"""Greedy generation with an uncached baseline and per-request KV caching."""
import argparse
import time

import torch

from engine.config import EngineConfig
from engine.model import GPT2Model
from engine.kv_cache import SimpleKVCache
from engine.sampler import greedy
from engine.weights import load_model


@torch.inference_mode()
def generate(model: GPT2Model, input_ids: torch.Tensor, max_new_tokens: int,
             *, eos_token_id: int | None = None, use_cache: bool = False,
             step_times: list[float] | None = None) -> torch.Tensor:
    """Return prompt plus generated IDs for one unpadded request.

    With eos_token_id=None, decode exactly max_new_tokens. Otherwise include
    EOS and stop. Reserve the entire requested output budget before decoding.
    Optional step_times collects complete token-step durations for benchmarks;
    its first duration includes validation/allocation/prefill. CUDA is synchronized
    only when collecting timings. Ordinary generation performs no timing calls.
    """
    device = model.token_embedding.weight.device
    if step_times is not None:
        if step_times:
            raise ValueError("step_times must be empty at request entry")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
    model.validate_input_ids(input_ids)
    if input_ids.shape[0] != 1:
        raise ValueError("Baseline generation accepts exactly one request")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if input_ids.shape[1] + max_new_tokens > model.config.max_positions:
        raise ValueError("Prompt plus output budget exceeds the model context limit")
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("eos_token_id must be inside the vocabulary")
    if max_new_tokens == 0:
        return input_ids
    cache = (SimpleKVCache(model.config, batch_size=1,
        capacity=input_ids.shape[1] + max_new_tokens, device=device,
        dtype=model.token_embedding.weight.dtype) if use_cache else None)
    output = input_ids
    for step in range(max_new_tokens):
        # Prefill once, then feed only the previous sampled token. The final
        # output token is never forwarded: there is no next prediction to make.
        current = output[:, -1:] if use_cache and step > 0 else output
        logits = model(current, cache=cache) if use_cache else model(current)
        token = greedy(logits[:, -1, :])
        # Only token IDs survive the step; retaining full logits doubles overlap
        # with the next forward, especially for a long uncached prefix.
        del logits
        output = torch.cat((output, token[:, None]), dim=1)
        if step_times is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            finished = time.perf_counter()
            step_times.append(finished - started)
            started = finished
        if eos_token_id is not None and token.item() == eos_token_id:
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer: GPT-2 generation with optional KV cache")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--use-cache", action="store_true", help="Prefill once, then decode with KV cache")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--int8", action="store_true", help="Store transformer projection weights as int8")
    args = parser.parse_args()
    try:
        if args.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        model, tokenizer = load_model(EngineConfig(device=args.device, int8=args.int8))
        device = model.token_embedding.weight.device
        # GPT-2 cannot forward an empty sequence; EOS is also its BOS seed.
        ids = (tokenizer(args.prompt, return_tensors="pt")["input_ids"] if args.prompt
               else torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long))
        output = generate(model, ids.to(device), args.max_new_tokens,
                          eos_token_id=tokenizer.eos_token_id, use_cache=args.use_cache)
    except ValueError as error:
        parser.error(str(error))
    # Preserve user text verbatim, including literal special-token spellings.
    # Only newly generated special tokens (including the terminal EOS) are hidden.
    new_ids = output[0, ids.shape[1]:].tolist()
    continuation = tokenizer.decode(new_ids, skip_special_tokens=True) if new_ids else ""
    print(args.prompt + continuation)


if __name__ == "__main__":
    main()
