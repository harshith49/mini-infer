"""Mixed request lengths: fixed FIFO cohorts versus continuous batching."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import torch

from benchmarks.bench_stages import hardware_name
from engine.batching import generate_batch
from engine.config import EngineConfig
from engine.generate import generate
from engine.model import GPT2Model
from engine.scheduler import Scheduler
from engine.weights import load_model


@torch.inference_mode()
def _run_once(model: GPT2Model, prompts: list[torch.Tensor], budgets: list[int], *,
              pad_token_id: int, max_batch_size: int, continuous: bool
              ) -> tuple[list[torch.Tensor], list[float], float, int, int | None]:
    device = model.token_embedding.weight.device
    def synchronize() -> None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    completions = [0.0] * len(prompts)
    peak_kv = 0
    if continuous:
        scheduler = Scheduler(model, max_batch_size=max_batch_size, pad_token_id=pad_token_id)
        for i, (prompt, budget) in enumerate(zip(prompts, budgets)):
            scheduler.submit(str(i), prompt, budget)
        while not scheduler.idle:
            events = scheduler.step()
            synchronize()
            delivered = time.perf_counter() - started
            for event in events:
                if event.finish_reason is not None:
                    completions[int(event.request_id)] = delivered
        outputs = [scheduler.result(str(i)) for i in range(len(prompts))]
        peak_kv = scheduler.peak_kv_bytes
    else:
        def observe_cache(module, args, kwargs):
            nonlocal peak_kv
            cache = kwargs.get('cache')
            if cache is not None:
                peak_kv = max(peak_kv, cache.allocated_bytes)
        hook = model.register_forward_pre_hook(observe_cache, with_kwargs=True)
        outputs = []
        try:
            for start in range(0, len(prompts), max_batch_size):
                cohort = prompts[start:start + max_batch_size]
                cohort_budgets = budgets[start:start + max_batch_size]
                generated = generate_batch(model, cohort, max(cohort_budgets),
                    pad_token_id=pad_token_id, use_cache=True)
                outputs.extend(output[:len(prompt) + budget].clone()
                    for output, prompt, budget in zip(generated, cohort, cohort_budgets))
                synchronize()
                delivered = time.perf_counter() - started
                completions[start:start + len(cohort)] = [delivered] * len(cohort)
        finally:
            hook.remove()
    synchronize()
    elapsed = time.perf_counter() - started
    process_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    return outputs, completions, elapsed, peak_kv, process_peak


def run_scheduler_workload(model: GPT2Model, prompts: list[torch.Tensor], budgets: list[int], *,
                           pad_token_id: int, max_batch_size: int, repetitions: int,
                           continuous: bool) -> dict[str, object]:
    """Include submission/allocation/copies and report useful-token throughput.

    Completion latency begins at common submission, ending at cohort return or
    delivery of a scheduler completion event. CPU process memory is unmeasured.
    """
    if (type(repetitions) is not int or repetitions < 3 or not prompts
            or len(prompts) != len(budgets)
            or any(type(budget) is not int or budget <= 0 for budget in budgets)):
        raise ValueError('Use matching nonempty prompts/positive budgets and at least three repetitions')
    validation = Scheduler(model, max_batch_size=max_batch_size, pad_token_id=pad_token_id)
    for i, (prompt, budget) in enumerate(zip(prompts, budgets)):
        validation.submit(str(i), prompt, budget)
    for start in range(0, len(prompts), max_batch_size):
        if (max(len(p) for p in prompts[start:start + max_batch_size])
                + max(budgets[start:start + max_batch_size]) > model.config.max_positions):
            raise ValueError('Padded static cohort plus output exceeds model context')
    del validation
    expected = [generate(model, prompt[None], budget, use_cache=True)[0]
                for prompt, budget in zip(prompts, budgets)]
    options = dict(pad_token_id=pad_token_id, max_batch_size=max_batch_size, continuous=continuous)
    elapsed, completions, peaks, process_peaks = [], [], [], []
    # One untimed warmup also verifies the comparator before measured repetitions.
    for repeat in range(repetitions + 1):
        outputs, finished, seconds, peak, process_peak = _run_once(model, prompts, budgets, **options)
        if any(not torch.equal(actual, want) for actual, want in zip(outputs, expected)):
            raise RuntimeError('Batch output differs from independent cached generation')
        if repeat:
            elapsed.append(seconds)
            completions.extend(finished)
            peaks.append(peak)
            if process_peak is not None:
                process_peaks.append(process_peak)
    useful = sum(budgets)
    excess = (0 if continuous else sum(
        len(budgets[start:start + max_batch_size]) * max(budgets[start:start + max_batch_size])
        - sum(budgets[start:start + max_batch_size]) for start in range(0, len(budgets), max_batch_size)))
    median = statistics.median(elapsed)
    percentiles = statistics.quantiles(completions, n=100, method='inclusive')
    device = model.token_embedding.weight.device
    return {
        'stage': 'continuous' if continuous else 'static_cohorts', 'device': str(device),
        'hardware': hardware_name(device), 'torch_version': torch.__version__,
        'threads': torch.get_num_threads(), 'request_count': len(prompts),
        'max_batch_size': max_batch_size, 'prompt_lengths': json.dumps([len(p) for p in prompts]),
        'output_budgets': json.dumps(budgets), 'repetitions': repetitions,
        'useful_generated_tokens': useful, 'extra_static_tokens': excess,
        'median_seconds': median, 'tokens_per_second': useful / median,
        'completion_p50_ms': percentiles[49] * 1000, 'completion_p95_ms': percentiles[94] * 1000,
        'peak_kv_bytes': max(peaks), 'peak_memory_bytes': max(process_peaks) if process_peaks else None,
        'peak_memory_kind': 'cuda_allocated_including_model' if process_peaks else 'unmeasured',
        'workload': 'synthetic token IDs; per-request budgets; greedy; seed=0',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--prompt-lengths', nargs='+', type=int, default=[16, 64, 32, 128] * 2)
    parser.add_argument('--output-budgets', nargs='+', type=int, default=[4, 32, 8, 64] * 2)
    parser.add_argument('--max-batch-size', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('results/continuous_batching.csv'))
    args = parser.parse_args()
    if (args.threads <= 0 or args.max_batch_size <= 0 or args.repetitions < 3
            or len(args.prompt_lengths) != len(args.output_budgets)
            or any(length <= 0 for length in args.prompt_lengths + args.output_budgets)):
        parser.error('Use matching positive length lists, positive threads/batch size and at least three repetitions')
    torch.set_num_threads(args.threads)
    try:
        model, _ = load_model(EngineConfig(device=args.device))
        generator = torch.Generator(device='cpu').manual_seed(0)
        prompts = [torch.randint(model.config.vocab_size, (length,), generator=generator).to(
            model.token_embedding.weight.device) for length in args.prompt_lengths]
        rows = []
        for continuous in (False, True):
            row = run_scheduler_workload(model, prompts, args.output_budgets,
                pad_token_id=0, max_batch_size=args.max_batch_size,
                repetitions=args.repetitions, continuous=continuous)
            rows.append(row)
            print(f"{row['stage']}: {row['tokens_per_second']:.2f} useful tokens/s", flush=True)
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
