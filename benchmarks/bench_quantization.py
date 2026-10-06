"""Int8 transformer weights: verified public text and paired CPU/CUDA measurements."""
import argparse
import csv
from dataclasses import asdict
import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import torch
from torch.nn import functional as F

from engine.model import GPT2Model
from engine.config import EngineConfig
from engine.weights import load_model
from engine.quantize import Int8Linear, quantize_model, model_storage_bytes
from benchmarks.bench_stages import run_workload

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


def run_comparison(model: GPT2Model, tokenizer, *, text: str, prompt_lengths: list[int],
                   max_new_tokens: int = 32, repetitions: int = 3, target_tokens: int = 4096,
                   context: int = 1024, stride: int = 512) -> list[dict[str, object]]:
    """Measure FP32, then convert the same exclusively owned model in place."""
    if (not isinstance(model, GPT2Model) or model.training
            or any(isinstance(layer, Int8Linear) for layer in model.modules())
            or any(parameter.dtype != torch.float32 for parameter in model.parameters())
            or not prompt_lengths or any(type(n) is not int or n <= 0 for n in prompt_lengths)
            or type(max_new_tokens) is not int or max_new_tokens <= 0
            or any(n + max_new_tokens > model.config.max_positions for n in prompt_lengths)
            or type(repetitions) is not int or repetitions < 3
            or type(target_tokens) is not int or target_tokens <= 0
            or type(context) is not int or not 2 <= context <= model.config.max_positions
            or type(stride) is not int or not 1 <= stride < context):
        raise ValueError('Comparison needs a fresh FP32 eval model and valid positive workloads/context/stride')
    _verified_text(text.encode('utf-8'))
    quality_ids = quality_input_ids(tokenizer, text, target_tokens=target_tokens)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompts = [torch.randint(model.config.vocab_size, (1, length), generator=generator)
               for length in prompt_lengths]
    device = model.token_embedding.weight.device
    rows, baseline_rates = [], []
    baseline_quality, baseline_bytes = None, None
    for quantized in (False, True):
        if quantized:
            quantize_model(model)
        quality = evaluate_perplexity(model, quality_ids, context=context, stride=stride)
        storage = model_storage_bytes(model)
        if not quantized:
            baseline_quality, baseline_bytes = quality, storage['weight_bytes']
        delta = quality['perplexity'] / baseline_quality['perplexity'] - 1
        if delta > .05:
            raise ValueError(f'Quantized perplexity increase {delta:.2%} exceeds the 5% quality gate')
        for i, prompt in enumerate(prompts):
            row = run_workload(model, prompt.to(device), max_new_tokens=max_new_tokens,
                               repetitions=repetitions, use_cache=True)
            if not quantized:
                baseline_rates.append(row['tokens_per_second'])
            row.update(storage)
            row.update(quality)
            row.update({
                'stage': 'int8_cached' if quantized else 'fp32_cached',
                'model_config': json.dumps(asdict(model.config)),
                'quantization_scope': ('48 GPT-2 block projections; embeddings/tied head remain FP32'
                                       if quantized and model.config.num_layers == 12 else
                                       'block projections; embeddings/tied head remain FP32' if quantized else 'FP32'),
                'quantization_formula': 's=max(abs(row))/127; round/clamp [-127,127]; FP32 dequant matmul' if quantized else '',
                'throughput_ratio': row['tokens_per_second'] / baseline_rates[i],
                'weight_reduction_fraction': 1 - storage['weight_bytes'] / baseline_bytes,
                'source_url': SAMPLE_URL, 'source_revision': SAMPLE_REVISION,
                'source_sha256': SAMPLE_SHA256,
                'token_sha256': hashlib.sha256(quality_ids.numpy().tobytes()).hexdigest(),
                'quality_context': context, 'quality_stride': stride,
                'fp32_nll': baseline_quality['nll'], 'fp32_perplexity': baseline_quality['perplexity'],
                'relative_ppl_delta': delta,
            })
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--prompt-lengths', nargs='+', type=int, default=[16,64,128,256])
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--quality-tokens', type=int, default=4096)
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--output', type=Path, default=Path('results/quantization.csv'))
    args = parser.parse_args()
    if (args.threads <= 0 or args.max_new_tokens <= 0 or args.repetitions < 3 or args.quality_tokens <= 0
            or any(n <= 0 for n in args.prompt_lengths) or not 1 <= args.stride < args.context):
        parser.error('Use positive threads/token counts, at least three repetitions and 1 <= stride < context')
    torch.set_num_threads(args.threads)
    try:
        model, tokenizer = load_model(EngineConfig(device=args.device))
        rows = run_comparison(model, tokenizer, text=load_quality_text(), prompt_lengths=args.prompt_lengths,
            max_new_tokens=args.max_new_tokens, repetitions=args.repetitions,
            target_tokens=args.quality_tokens, context=args.context, stride=args.stride)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['stage']}: prompt={row['prompt_tokens']}, {row['tokens_per_second']:.2f} tokens/s, "
              f"weights={row['weight_bytes']/1024**2:.2f} MiB, PPL={row['perplexity']:.4f}")
    print(f'Saved {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
