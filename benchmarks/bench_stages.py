"""Naive versus contiguous KV cache; actual timings, explicit memory scopes."""
import argparse
import csv
import platform
from pathlib import Path
import statistics
import subprocess
import time

import torch

from engine.config import EngineConfig
from engine.generate import generate
from engine.kv_cache import SimpleKVCache
from engine.model import GPT2Model
from engine.weights import load_model


def hardware_name(device: torch.device) -> str:
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    cpu = platform.processor() or platform.machine()
    if platform.system() == 'Darwin':
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            cpu = result.stdout.strip()
    return f'{cpu}; {platform.platform()}'


def run_workload(model: GPT2Model, input_ids: torch.Tensor, *, max_new_tokens: int,
                 repetitions: int, use_cache: bool) -> dict[str, object]:
    """Measure one stage, including allocation and prefill; EOS is disabled.

    CPU peak process memory is deliberately unmeasured. CUDA peak counts all
    PyTorch allocated tensors, including resident model weights, not process RAM.
    """
    if (not isinstance(repetitions, int) or repetitions < 3
            or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
        raise ValueError('Use positive output length and at least three repetitions')
    device = model.token_embedding.weight.device
    warmup = generate(model, input_ids, max_new_tokens, use_cache=use_cache)
    elapsed, first_tokens, decode_times, peaks = [], [], [], []
    for _ in range(repetitions):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        steps: list[float] = []
        started = time.perf_counter()
        output = generate(model, input_ids, max_new_tokens,
                          use_cache=use_cache, step_times=steps)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - started)
        first_tokens.append(steps[0])
        decode_times.extend(steps[1:])
        if device.type == 'cuda':
            peaks.append(torch.cuda.max_memory_allocated(device))
        if output.shape[1] != input_ids.shape[1] + max_new_tokens or not torch.equal(output, warmup):
            raise RuntimeError('Benchmark output count or deterministic output changed')
    median = statistics.median(elapsed)
    percentiles = statistics.quantiles(decode_times, n=100, method='inclusive') if decode_times else None
    # Inspect real tensor allocation outside the measured intervals. Do not label
    # reserved request bytes as peak process memory or include them in the timers.
    cache_bytes = (SimpleKVCache(model.config, batch_size=1,
        capacity=input_ids.shape[1] + max_new_tokens, device=device,
        dtype=model.token_embedding.weight.dtype).allocated_bytes if use_cache else 0)
    return {
        'stage': 'kv_cache' if use_cache else 'naive', 'device': str(device),
        'hardware': hardware_name(device), 'torch_version': torch.__version__,
        'threads': torch.get_num_threads(), 'prompt_tokens': input_ids.shape[1],
        'generated_tokens': max_new_tokens, 'repetitions': repetitions,
        'median_seconds': median, 'tokens_per_second': max_new_tokens / median,
        'ttft_seconds': statistics.median(first_tokens),
        'decode_p50_ms': percentiles[49] * 1000 if percentiles else None,
        'decode_p95_ms': percentiles[94] * 1000 if percentiles else None,
        'cache_allocated_bytes': cache_bytes,
        'peak_memory_bytes': max(peaks) if peaks else None,
        'peak_memory_kind': 'cuda_allocated_including_model' if peaks else 'unmeasured',
        'workload': 'synthetic token IDs; fixed output; greedy; seed=0',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--prompt-lengths', nargs='+', type=int, default=[16, 64, 128, 256])
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('results/kv_cache.csv'))
    args = parser.parse_args()
    if args.threads <= 0 or args.max_new_tokens <= 0 or args.repetitions < 3:
        parser.error('Use positive threads/output length and at least three repetitions')
    if any(length <= 0 for length in args.prompt_lengths):
        parser.error('Prompt lengths must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    try:
        model, _ = load_model(EngineConfig(device=args.device))
        if any(length + args.max_new_tokens > model.config.max_positions for length in args.prompt_lengths):
            raise ValueError('Prompt plus output exceeds context capacity')
        rows = []
        for length in args.prompt_lengths:
            ids = torch.randint(model.config.vocab_size, (1, length), device='cpu')
            ids = ids.to(model.token_embedding.weight.device)
            # Correctness gate precedes timing; never publish incomparable workloads.
            if not torch.equal(generate(model, ids, args.max_new_tokens),
                               generate(model, ids, args.max_new_tokens, use_cache=True)):
                raise RuntimeError(f'Cached/naive output mismatch for prompt length {length}')
            for cached in (False, True):
                row = run_workload(model, ids, max_new_tokens=args.max_new_tokens,
                                   repetitions=args.repetitions, use_cache=cached)
                rows.append(row)
                print(f"{row['stage']}: prompt={length}, {row['tokens_per_second']:.2f} tokens/s", flush=True)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
