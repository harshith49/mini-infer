"""Fixed-batch greedy generation with left padding and optional KV caching."""
import argparse
import json

import torch

from engine.config import EngineConfig
from engine.kv_cache import SimpleKVCache
from engine.model import GPT2Model
from engine.sampler import greedy
from engine.weights import load_model


@torch.inference_mode()
def generate_batch(model: GPT2Model, prompts: list[torch.Tensor], max_new_tokens: int,
                   *, pad_token_id: int, eos_token_id: int | None = None,
                   use_cache: bool = False) -> list[torch.Tensor]:
    """Return unpadded prompt plus continuation for each row, in input order.

    A shared output budget reserves physical slots for the longest prompt.
    Finished rows keep their EOS as a valid key and mask later filler tokens.
    """
    device = model.token_embedding.weight.device
    if not prompts:
        raise ValueError('prompts must contain at least one request')
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    for name, token in [('pad_token_id', pad_token_id), ('eos_token_id', eos_token_id)]:
        if name == 'eos_token_id' and token is None:
            continue
        if not isinstance(token, int) or not 0 <= token < model.config.vocab_size:
            raise ValueError(f'{name} must be an integer inside the vocabulary')
    for prompt in prompts:
        if not isinstance(prompt, torch.Tensor) or prompt.ndim != 1 or prompt.device != device:
            raise ValueError('Each prompt must be a rank-1 tensor on the model device')
        model.validate_input_ids(prompt[None])
    width = max(len(prompt) for prompt in prompts)
    if width + max_new_tokens > model.config.max_positions:
        raise ValueError('Longest prompt plus output budget exceeds the model context limit')
    if max_new_tokens == 0:
        return prompts

    batch = len(prompts)
    output = torch.full((batch, width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
    for row, prompt in enumerate(prompts):
        output[row, -len(prompt):] = prompt
        mask[row, -len(prompt):] = True
    cache = (SimpleKVCache(model.config, batch_size=batch, capacity=width + max_new_tokens,
        device=device, dtype=model.token_embedding.weight.dtype) if use_cache else None)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    counts = torch.zeros(batch, dtype=torch.long, device=device)
    # ponytail: finished rows retain compute/cache slots; compact them in the scheduler milestone.
    for step in range(max_new_tokens):
        current = output[:, -1:] if use_cache and step > 0 else output
        logits = model(current, cache=cache, attention_mask=mask)
        tokens = greedy(logits[:, -1, :])
        del logits
        tokens = tokens.masked_fill(~active, pad_token_id)
        output = torch.cat((output, tokens[:, None]), dim=1)
        mask = torch.cat((mask, active[:, None]), dim=1)
        counts += active
        if eos_token_id is not None:
            active = active & (tokens != eos_token_id)
            if not active.any():
                break
    return [output[row, width - len(prompt):width + count]
            for row, (prompt, count) in enumerate(zip(prompts, counts.tolist()))]


def main() -> None:
    parser = argparse.ArgumentParser(description='mini-infer: fixed-batch GPT-2 generation')
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=50)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument("--int8", action="store_true", help="Store transformer projection weights as int8")
    args = parser.parse_args()
    try:
        if args.max_new_tokens < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer')
        model, tokenizer = load_model(EngineConfig(device=args.device, int8=args.int8))
        device = model.token_embedding.weight.device
        prompts = [(tokenizer(text, return_tensors='pt')['input_ids'][0] if text
                    else torch.tensor([tokenizer.eos_token_id], dtype=torch.long)).to(device)
                   for text in args.prompt]
        outputs = generate_batch(model, prompts, args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=args.use_cache)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps([text + tokenizer.decode(output[len(prompt):].tolist(), skip_special_tokens=True)
                      for text, prompt, output in zip(args.prompt, prompts, outputs)], ensure_ascii=False))


if __name__ == '__main__':
    main()
