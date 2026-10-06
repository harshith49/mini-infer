"""Fixed KV budget: real reservation capacity and labelled arena fragmentation."""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from benchmarks.bench_stages import hardware_name
from engine.config import EngineConfig, ModelConfig
from engine.generate import generate
from engine.kv_cache import PagePool, PagedKVCache, SimpleKVCache
from engine.scheduler import Scheduler
from engine.weights import load_model


class _Arena:
    """Allocation-only first-fit intervals; no PyTorch allocator claim."""

    def __init__(self, slots: int) -> None:
        self._free = [(0, slots)]
        self.live: dict[str, tuple[int, int]] = {}

    @property
    def free_slots(self) -> int:
        return sum(length for _, length in self._free)

    @property
    def largest_free_run(self) -> int:
        return max((length for _, length in self._free), default=0)

    def allocate(self, name: str, capacity: int) -> bool:
        if name in self.live or type(capacity) is not int or capacity <= 0:
            raise ValueError('Arena allocation requires unique ID and positive capacity')
        for i, (start, length) in enumerate(self._free):
            if capacity <= length:
                self.live[name] = (start, capacity)
                self._free[i:i+1] = [(start + capacity, length - capacity)] if capacity < length else []
                return True
        return False

    def release(self, name: str) -> None:
        block = self.live.pop(name)
        merged = []
        for start, length in sorted(self._free + [block]):
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self._free = merged


def _budget(config: ModelConfig, budget_bytes: int, page_size: int,
            device: torch.device, dtype: torch.dtype) -> tuple[int, int, int, torch.device]:
    if (not isinstance(config, ModelConfig) or type(budget_bytes) is not int or budget_bytes <= 0
            or type(page_size) is not int or page_size <= 0
            or not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
        raise ValueError('Use model dimensions, positive integer budget/page size and floating dtype')
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError('Invalid benchmark device') from error
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Benchmark needs available CPU or CUDA')
    token_bytes = 2 * config.num_layers * config.hidden_size * (torch.finfo(dtype).bits // 8)
    pages = budget_bytes // (token_bytes * page_size)
    if pages < 1:
        raise ValueError('Budget cannot hold one KV page')
    return token_bytes, pages, pages * token_bytes * page_size, device


def _row(config, *, stage, experiment, allocation_kind, budget_bytes, page_size, device,
         dtype, effective, capacities, trace, live, resident, reserved, rounding=0,
         before_free=None, before_largest=None, after_free=None, after_largest=None,
         probe=None) -> dict[str, object]:
    return {
        'stage': stage, 'experiment': experiment, 'allocation_kind': allocation_kind,
        'device': str(device), 'hardware': hardware_name(device), 'torch_version': torch.__version__,
        'dtype': str(dtype), 'model_config': json.dumps(asdict(config)), 'page_size': page_size, 'requested_budget_bytes': budget_bytes,
        'effective_budget_bytes': effective, 'budget_remainder_bytes': budget_bytes - effective,
        'request_capacities': json.dumps(capacities), 'allocation_trace': json.dumps(trace),
        'live_requests': live, 'probe_admitted': probe, 'resident_bytes': resident,
        'reserved_bytes': reserved, 'used_bytes': 0, 'rounding_bytes': rounding,
        'unused_logical_bytes': reserved - rounding,
        'free_slots_before_probe': before_free, 'largest_free_run_before_probe': before_largest,
        'free_slots_after_probe': after_free, 'largest_free_run_after_probe': after_largest,
        'peak_memory_bytes': None, 'peak_memory_kind': 'unmeasured',
    }


def run_capacity_experiment(config: ModelConfig, *, capacities: list[int], budget_bytes: int,
                            page_size: int, device: torch.device, dtype: torch.dtype
                            ) -> list[dict[str, object]]:
    if (not isinstance(config, ModelConfig) or not capacities
            or any(type(n) is not int or not 0 < n <= config.max_positions for n in capacities)):
        raise ValueError('Capacities must be positive integers within model context')
    token_bytes, pages, effective, device = _budget(config, budget_bytes, page_size, device, dtype)
    rows = []
    for paged in (False, True):
        pool = PagePool(config, num_pages=pages, page_size=page_size, device=device, dtype=dtype) if paged else None
        caches, trace = [], []
        cache = None
        reserved = 0
        try:
            for i, capacity in enumerate(capacities):
                needed = ((capacity + page_size - 1) // page_size * page_size if paged else capacity) * token_bytes
                if reserved + needed > effective:
                    trace.append({'op': 'allocate', 'id': str(i), 'capacity': capacity, 'accepted': False})
                    break
                cache = (PagedKVCache(pool, capacity=capacity) if paged else SimpleKVCache(config,
                    batch_size=1, capacity=capacity, device=device, dtype=dtype))
                caches.append(cache)
                reserved += cache.allocated_bytes
                entry = {'op': 'allocate', 'id': str(i), 'capacity': capacity, 'accepted': True}
                if paged:
                    entry['page_ids'] = cache.page_ids
                trace.append(entry)
            rows.append(_row(config, stage='paged' if paged else 'contiguous', experiment='clean_capacity',
                allocation_kind='real_page_pool' if paged else 'real_contiguous_tensors', budget_bytes=budget_bytes,
                page_size=page_size, device=device, dtype=dtype, effective=effective, capacities=capacities,
                trace=trace, live=len(caches), resident=pool.allocated_bytes if paged else reserved,
                reserved=reserved, rounding=sum(cache.rounding_bytes for cache in caches) if paged else 0))
        finally:
            for cache in caches:
                if paged:
                    cache.close()
            caches.clear()
            cache = None
        if pool is not None and pool.free_pages != pool.num_pages:
            raise RuntimeError('Capacity experiment leaked page reservations')
    return rows


def _largest_page_run(pool: PagePool) -> int:
    previous, current, largest = -2, 0, 0
    for page in pool.free_page_ids:
        current = current + 1 if page == previous + 1 else 1
        largest = max(largest, current)
        previous = page
    return largest * pool.page_size


def run_fragmentation_experiment(config: ModelConfig, *, budget_bytes: int, page_size: int,
                                 device: torch.device, dtype: torch.dtype) -> list[dict[str, object]]:
    token_bytes, pages, effective, device = _budget(config, budget_bytes, page_size, device, dtype)
    if pages < 4 or 2 * page_size > config.max_positions:
        raise ValueError('Fragmentation trace needs four pages and a two-page request within context')
    capacities = [page_size] * pages + [2 * page_size]
    rows = []
    for paged in (False, True):
        pool = PagePool(config, num_pages=pages, page_size=page_size, device=device, dtype=dtype) if paged else None
        arena = _Arena(pages * page_size) if not paged else None
        live: dict[str, PagedKVCache] = {}
        trace = []
        try:
            for i in range(pages):
                name = str(i)
                if paged:
                    live[name] = PagedKVCache(pool, capacity=page_size)
                elif not arena.allocate(name, page_size):
                    raise RuntimeError('Initial arena fill failed')
                entry = {'op': 'allocate', 'id': name, 'capacity': page_size, 'accepted': True}
                if paged:
                    entry['page_ids'] = live[name].page_ids
                trace.append(entry)
            for i in range(0, pages, 2):
                name = str(i)
                if paged:
                    live.pop(name).close()
                else:
                    arena.release(name)
                trace.append({'op': 'free', 'id': name})
            before_free = pool.free_pages * page_size if paged else arena.free_slots
            before_largest = _largest_page_run(pool) if paged else arena.largest_free_run
            if paged:
                live['probe'] = PagedKVCache(pool, capacity=2 * page_size)
                admitted = True
            else:
                admitted = arena.allocate('probe', 2 * page_size)
            entry = {'op': 'allocate', 'id': 'probe', 'capacity': 2 * page_size, 'accepted': admitted}
            if paged:
                entry['page_ids'] = live['probe'].page_ids
            trace.append(entry)
            reserved = (sum(cache.allocated_bytes for cache in live.values()) if paged
                        else sum(capacity for _, capacity in arena.live.values()) * token_bytes)
            rows.append(_row(config, stage='paged' if paged else 'contiguous', experiment='fragmentation_trace',
                allocation_kind='real_page_pool' if paged else 'metadata_first_fit_arena', budget_bytes=budget_bytes,
                page_size=page_size, device=device, dtype=dtype, effective=effective, capacities=capacities,
                trace=trace, live=len(live) if paged else len(arena.live), resident=pool.allocated_bytes if paged else None,
                reserved=reserved, before_free=before_free, before_largest=before_largest,
                after_free=pool.free_pages * page_size if paged else arena.free_slots,
                after_largest=_largest_page_run(pool) if paged else arena.largest_free_run, probe=admitted))
        finally:
            for cache in live.values():
                cache.close()
        if pool is not None and pool.free_pages != pool.num_pages:
            raise RuntimeError('Fragmentation experiment leaked pages')
    return rows


def check_inference(model, prompts: list[torch.Tensor], budgets: list[int], *, num_pages: int,
                    page_size: int, max_batch_size: int, pad_token_id: int) -> None:
    expected = [generate(model, prompt[None], budget, use_cache=True)[0]
                for prompt, budget in zip(prompts, budgets)]
    for paged in (False, True):
        pool = PagePool(model.config, num_pages=num_pages, page_size=page_size,
            device=model.token_embedding.weight.device, dtype=model.token_embedding.weight.dtype) if paged else None
        scheduler = Scheduler(model, max_batch_size=max_batch_size, pad_token_id=pad_token_id, page_pool=pool)
        try:
            for i, (prompt, budget) in enumerate(zip(prompts, budgets)):
                scheduler.submit(str(i), prompt, budget)
            while not scheduler.idle:
                scheduler.step()
            if any(not torch.equal(scheduler.result(str(i)), output) for i, output in enumerate(expected)):
                raise RuntimeError('Batch output differs from independent cached generation')
        finally:
            for request in scheduler._requests.values():
                if isinstance(request.cache, PagedKVCache):
                    request.cache.close()
                    request.cache = None
        if pool is not None and pool.free_pages != pool.num_pages:
            raise RuntimeError('Inference gate leaked pages')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--budget-mib', type=float, default=32)
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--capacities', nargs='+', type=int, default=[17] * 32)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('results/paged_cache.csv'))
    args = parser.parse_args()
    if (not math.isfinite(args.budget_mib) or args.budget_mib <= 0 or args.page_size <= 0
            or args.threads <= 0 or any(capacity <= 0 for capacity in args.capacities)):
        parser.error('Use finite positive budget, capacities, page size and threads')
    torch.set_num_threads(args.threads)
    try:
        model, _ = load_model(EngineConfig(device=args.device))
        device, dtype = model.token_embedding.weight.device, model.token_embedding.weight.dtype
        budget_bytes = int(args.budget_mib * 1024 * 1024)
        _, pages, _, _ = _budget(model.config, budget_bytes, args.page_size, device, dtype)
        if pages < 4 or 2 * args.page_size > model.config.max_positions:
            raise ValueError('Budget/page size cannot support the fragmentation trace')
        if any(capacity > model.config.max_positions for capacity in args.capacities):
            raise ValueError('Request capacity exceeds model context')
        generator = torch.Generator(device='cpu').manual_seed(0)
        prompts = [torch.randint(model.config.vocab_size, (length,), generator=generator).to(device)
                   for length in [16, 64, 32, 128] * 2]
        check_inference(model, prompts, [4, 32, 8, 64] * 2, num_pages=pages,
                        page_size=args.page_size, max_batch_size=2, pad_token_id=0)
        rows = run_capacity_experiment(model.config, capacities=args.capacities,
            budget_bytes=budget_bytes, page_size=args.page_size, device=device, dtype=dtype)
        rows += run_fragmentation_experiment(model.config, budget_bytes=budget_bytes,
            page_size=args.page_size, device=device, dtype=dtype)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['experiment']} {row['stage']}: live={row['live_requests']}, probe={row['probe_admitted']}")
    print(f'Saved {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
