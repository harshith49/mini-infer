"""One workload, six engine stages, and explicit measurement scopes."""
import argparse
from collections.abc import Callable
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import torch

from benchmarks.bench_stages import hardware_name
from benchmarks.load_test import percentile
from engine.batching import generate_batch
from engine.config import ModelConfig
from engine.generate import generate
from engine.kv_cache import PagePool
from engine.model import GPT2Model
from engine.quantize import Int8Linear, model_storage_bytes, quantize_model
from engine.scheduler import Scheduler

STAGES = ('naive', 'kv_cache', 'static_batching', 'continuous_batching', 'paged_kv', 'int8')
DEFAULT_LENGTHS = (16, 64, 32, 128) * 2
DEFAULT_BUDGETS = (4, 32, 8, 64) * 2


@dataclass(frozen=True)
class Workload:
    prompt_lengths: tuple[int, ...] = DEFAULT_LENGTHS
    output_budgets: tuple[int, ...] = DEFAULT_BUDGETS
    max_batch_size: int = 2
    repetitions: int = 3
    seed: int = 0
    page_size: int = 16
    num_pages: int = 32

    def __post_init__(self):
        if (not self.prompt_lengths or len(self.prompt_lengths) != len(self.output_budgets)
                or any(type(n) is not int or n <= 0 for n in (*self.prompt_lengths,
                    *self.output_budgets, self.max_batch_size, self.page_size, self.num_pages))
                or type(self.repetitions) is not int or self.repetitions < 3
                or type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError('Use matching positive lengths/budgets, positive batch/pages, seed 0..2**63-1 and >=3 repetitions')


@dataclass
class RunResult:
    outputs: list[torch.Tensor]
    completions: list[float]
    elapsed: float
    peak_kv_bytes: int | None
    peak_memory_bytes: int | None


def make_prompts(config: ModelConfig, workload: Workload, device: torch.device) -> list[torch.Tensor]:
    for length, budget in zip(workload.prompt_lengths, workload.output_budgets):
        if length + budget > config.max_positions:
            raise ValueError('Prompt plus output exceeds context')
        if math.ceil((length + budget) / workload.page_size) > workload.num_pages:
            raise ValueError('Request exceeds entire paged pool')
    for start in range(0, len(workload.prompt_lengths), workload.max_batch_size):
        end = start + workload.max_batch_size
        if max(workload.prompt_lengths[start:end]) + max(workload.output_budgets[start:end]) > config.max_positions:
            raise ValueError('Padded static cohort exceeds context')
    rng = torch.Generator(device='cpu').manual_seed(workload.seed)
    return [torch.randint(config.vocab_size, (n,), generator=rng).to(device) for n in workload.prompt_lengths]


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_stage_once(model: GPT2Model, prompts: list[torch.Tensor], workload: Workload, stage: str) -> RunResult:
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    device = model.token_embedding.weight.device
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs, completions = [], [0.] * len(prompts)
    peak_kv = 0
    scheduler = None
    # Hooks observe allocations outside the model's actual arithmetic.
    def observe(module, args, kwargs):
        nonlocal peak_kv
        cache = kwargs.get('cache')
        if cache is not None:
            peak_kv = max(peak_kv, cache.allocated_bytes)
    hook = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        if stage in ('naive', 'kv_cache'):
            for i, (prompt, budget) in enumerate(zip(prompts, workload.output_budgets)):
                outputs.append(generate(model, prompt[None], budget, use_cache=stage == 'kv_cache')[0])
                synchronize(device)
                completions[i] = time.perf_counter() - started
        elif stage == 'static_batching':
            for start in range(0, len(prompts), workload.max_batch_size):
                cohort = prompts[start:start + workload.max_batch_size]
                budgets = workload.output_budgets[start:start + workload.max_batch_size]
                generated = generate_batch(model, cohort, max(budgets), pad_token_id=0, use_cache=True)
                outputs.extend(o[:len(p)+b].clone() for o,p,b in zip(generated,cohort,budgets))
                del generated
                synchronize(device)
                completions[start:start+len(cohort)] = [time.perf_counter()-started] * len(cohort)
        else:
            pool = (PagePool(model.config, num_pages=workload.num_pages, page_size=workload.page_size,
                    device=device, dtype=model.token_embedding.weight.dtype) if stage in ('paged_kv','int8') else None)
            scheduler = Scheduler(model, max_batch_size=workload.max_batch_size, pad_token_id=0, page_pool=pool)
            for i,(prompt,budget) in enumerate(zip(prompts,workload.output_budgets)):
                scheduler.submit(str(i),prompt,budget)
            while not scheduler.idle:
                events = scheduler.step()
                synchronize(device)
                delivered = time.perf_counter()-started
                for event in events:
                    if event.finish_reason is not None:
                        completions[int(event.request_id)] = delivered
            outputs = [scheduler.result(str(i)) for i in range(len(prompts))]
            peak_kv = scheduler.peak_kv_bytes
        synchronize(device)
        elapsed = time.perf_counter()-started
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        return RunResult(outputs,completions,elapsed,peak_kv,peak)
    finally:
        hook.remove()
        if scheduler is not None:
            scheduler.close()


def measure(run_once: Callable[[], RunResult], expected: list[torch.Tensor], workload: Workload) -> dict[str, object]:
    expected = [x.detach().cpu() for x in expected]
    elapsed, completions, kv, peaks = [], [], [], []
    for repeat in range(workload.repetitions+1):
        result = run_once()
        if (len(result.outputs) != len(expected) or len(result.completions) != len(expected)
                or any(not torch.equal(actual.cpu(), want) for actual,want in zip(result.outputs,expected))):
            raise RuntimeError('Benchmark output count or deterministic tokens changed')
        if (not math.isfinite(result.elapsed) or result.elapsed <= 0
                or any(not math.isfinite(t) or not 0 < t <= result.elapsed for t in result.completions)):
            raise RuntimeError('Invalid benchmark timing observations')
        if repeat:
            elapsed.append(result.elapsed)
            completions.extend(result.completions)
            if result.peak_kv_bytes is not None:
                kv.append(result.peak_kv_bytes)
            if result.peak_memory_bytes is not None:
                peaks.append(result.peak_memory_bytes)
        del result  # Do not retain GPU warmup/output tensors into the next peak interval.
    median = statistics.median(elapsed)
    useful = sum(workload.output_budgets)
    return dict(useful_generated_tokens=useful, median_seconds=median,
        tokens_per_second=useful/median, completion_p50_seconds=percentile(completions,.5),
        completion_p95_seconds=percentile(completions,.95), peak_kv_bytes=max(kv) if kv else None,
        peak_memory_bytes=max(peaks) if peaks else None,
        peak_memory_kind='cuda_allocated_including_model' if peaks else 'unmeasured')


def metadata(model: torch.nn.Module, prompts: list[torch.Tensor], workload: Workload, stage: str, device: torch.device) -> dict[str, object]:
    paged = stage in ('paged_kv','int8')
    extra = sum(len(workload.output_budgets[s:s+workload.max_batch_size]) * max(workload.output_budgets[s:s+workload.max_batch_size])
                - sum(workload.output_budgets[s:s+workload.max_batch_size]) for s in range(0,len(prompts),workload.max_batch_size)) if stage == 'static_batching' else 0
    return dict(stage=stage, device=str(device), hardware=hardware_name(device),
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        dtype='int8_weights_fp32_compute' if stage == 'int8' else 'fp32', threads=torch.get_num_threads(),
        prompt_lengths=json.dumps(workload.prompt_lengths), output_budgets=json.dumps(workload.output_budgets),
        request_count=len(prompts), max_batch_size=workload.max_batch_size, seed=workload.seed,
        warmups=1, repetitions=workload.repetitions, extra_static_tokens=extra,
        cache_backend='paged' if paged else 'none' if stage == 'naive' else 'contiguous',
        page_size=workload.page_size if paged else None, num_pages=workload.num_pages if paged else None,
        kv_memory_scope='pool_plus_workspace_and_live_gather' if paged else 'cache_plus_workspace' if stage == 'continuous_batching' else 'cache_reservation',
        latency_scope='common_submission_to_request_or_cohort_completion_seconds',
        workload='synthetic token IDs; fixed outputs; greedy; no EOS stop', **model_storage_bytes(model))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError('CSV rows must have matching nonempty schemas')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_suite(model: GPT2Model, tokenizer, workload: Workload, *, cache_dir: str = 'model_cache') -> list[dict[str, object]]:
    from benchmarks import bench_quantization as quality
    if (model.training or any(isinstance(m,Int8Linear) for m in model.modules())
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('Suite requires a fresh FP32 eval model')
    device = model.token_embedding.weight.device
    prompts = make_prompts(model.config,workload,device)
    expected = [generate(model,p[None],b,use_cache=True)[0].cpu() for p,b in zip(prompts,workload.output_budgets)]
    text = quality.load_quality_text(cache_dir)
    quality_ids = quality.quality_input_ids(tokenizer,text,target_tokens=4096)
    context = min(1024,model.config.max_positions)
    fp32 = quality.evaluate_perplexity(model,quality_ids,context=context,stride=context//2)
    rows = []
    for stage in STAGES:
        scored = fp32
        if stage == 'int8':
            quantize_model(model)
            scored = quality.evaluate_perplexity(model,quality_ids,context=context,stride=context//2)
            if scored['perplexity']/fp32['perplexity']-1 > .05:
                raise ValueError('Int8 quality gate exceeds 5% perplexity increase')
            result = run_stage_once(model,prompts,workload,stage)
            expected = [x.cpu() for x in result.outputs]
            del result
        row = metadata(model,prompts,workload,stage,device)
        row.update(measure(lambda: run_stage_once(model,prompts,workload,stage),expected,workload))
        row.update(scored)
        row.update(fp32_perplexity=fp32['perplexity'], relative_ppl_delta=scored['perplexity']/fp32['perplexity']-1,
            source_url=quality.SAMPLE_URL,source_revision=quality.SAMPLE_REVISION,source_sha256=quality.SAMPLE_SHA256,
            token_sha256=hashlib.sha256(quality_ids.numpy().tobytes()).hexdigest(),quality_context=context,quality_stride=context//2)
        rows.append(row)
        print(f"{stage}: {row['tokens_per_second']:.2f} useful tokens/s",flush=True)
    return rows


def arguments(description: str, default_output: str, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--prompt-lengths',nargs='+',type=int,default=DEFAULT_LENGTHS)
    parser.add_argument('--output-budgets',nargs='+',type=int,default=DEFAULT_BUDGETS)
    parser.add_argument('--max-batch-size',type=int,default=2)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--page-size',type=int,default=16)
    parser.add_argument('--num-pages',type=int,default=32)
    parser.add_argument('--output',type=Path,default=Path(default_output))
    args = parser.parse_args(argv)
    try:
        if args.threads <= 0:
            raise ValueError('Threads must be positive')
        workload = Workload(tuple(args.prompt_lengths),tuple(args.output_budgets),args.max_batch_size,args.repetitions,args.seed,args.page_size,args.num_pages)
    except ValueError as error:
        parser.error(str(error))
    return parser,args,workload


def main(argv=None):
    from engine.config import EngineConfig
    from engine.weights import load_model
    parser,args,workload = arguments(__doc__,'results/stages.csv',argv)
    torch.set_num_threads(args.threads)
    try:
        model,tokenizer = load_model(EngineConfig(device=args.device))
        rows = run_suite(model,tokenizer,workload)
        write_rows(args.output,rows)
    except ValueError as error:
        parser.error(str(error))
    print(f'Saved {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
