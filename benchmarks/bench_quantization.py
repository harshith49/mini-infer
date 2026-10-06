"""Int8 transformer weights: verified public text and paired CPU/CUDA measurements."""
import hashlib
import math
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import torch
from torch.nn import functional as F

from engine.model import GPT2Model

SAMPLE_REVISION = '370cbcd448eb7daf32f21a6be560b70e0b33c4e3'
SAMPLE_URL = f'https://raw.githubusercontent.com/karpathy/char-rnn/{SAMPLE_REVISION}/data/tinyshakespeare/input.txt'
SAMPLE_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
SAMPLE_BYTES = 1115394


def _verified_text(data: bytes) -> str:
    if len(data) != SAMPLE_BYTES or hashlib.sha256(data).hexdigest() != SAMPLE_SHA256:
        raise ValueError('Quality sample size/checksum mismatch; remove corrupt cache and download again')
    return data.decode('utf-8')


def load_quality_text(cache_dir: str | Path = 'model_cache') -> str:
    """Cache only complete, checksum-verified upstream text; respect offline mode."""
    destination = Path(cache_dir) / 'quality' / 'tinyshakespeare.txt'
    if destination.exists():
        with destination.open('rb') as handle:
            return _verified_text(handle.read(SAMPLE_BYTES + 1))
    if os.environ.get('HF_HUB_OFFLINE') == '1':
        raise OSError('Quality sample not cached in offline mode; run the quantization benchmark once online to download it')
    with urlopen(SAMPLE_URL, timeout=30) as response:
        data = response.read(SAMPLE_BYTES + 1)
    text = _verified_text(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return text


def quality_input_ids(tokenizer, text: str, *, target_tokens: int = 4096) -> torch.Tensor:
    """A deterministic CPU prefix; the first token is context, not a target."""
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError('Quality target count must be a positive integer')
    ids = tokenizer(text, add_special_tokens=False, verbose=False)['input_ids']
    if len(ids) < target_tokens + 1:
        raise ValueError('Quality sample does not contain enough tokens')
    return torch.tensor(ids[:target_tokens + 1], dtype=torch.long)


@torch.inference_mode()
def evaluate_perplexity(model: GPT2Model, input_ids: torch.Tensor, *, context: int = 1024,
                        stride: int = 512) -> dict[str, float | int]:
    """Score each next-token target once, with a bounded preceding context."""
    if (type(context) is not int or not 2 <= context <= model.config.max_positions
            or type(stride) is not int or not 1 <= stride < context
            or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1
            or input_ids.dtype != torch.long or len(input_ids) < 2
            or input_ids.device.type not in ('cpu', 'cuda')
            or torch.any(input_ids < 0) or torch.any(input_ids >= model.config.vocab_size)):
        raise ValueError('Quality needs rank-1 valid long IDs, context within model bounds and 1 <= stride < context')
    if model.training:
        raise ValueError('Quality evaluation requires an eval-mode model')
    device = model.token_embedding.weight.device
    total, count = 0., 0
    for a in range(1, len(input_ids), stride):
        b = min(a + stride, len(input_ids))
        start = max(0, b - context)
        tokens = input_ids[start:b].to(device)[None]
        logits = model(tokens)
        # The suffix contains new targets only; earlier overlapping context is never rescored.
        scores = logits[0, a-start-1:b-start-1].double()
        del logits
        loss = F.cross_entropy(scores, tokens[0, a-start:b-start], reduction='sum').item()
        del scores, tokens
        if not math.isfinite(loss):
            raise ValueError('Quality NLL must remain finite')
        total += loss
        count += b - a
    nll = total / count
    perplexity = math.exp(nll)
    if not math.isfinite(perplexity):
        raise ValueError('Quality perplexity must remain finite')
    return {'scored_tokens': count, 'nll': nll, 'perplexity': perplexity}
