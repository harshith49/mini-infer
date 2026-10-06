"""Capacity uses real allocations; fragmented contiguous arenas use metadata."""
import csv
import io
import json

import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model


@pytest.fixture
def config():
    return ModelConfig(vocab_size=37, max_positions=64, hidden_size=24,
                       num_layers=2, num_heads=4, intermediate_size=96)


def options():
    return {'budget_bytes': 6500, 'page_size': 4, 'device': torch.device('cpu'), 'dtype': torch.float32}


def test_clean_capacity_rounding_and_equal_effective_budgets(config):
    from benchmarks.bench_paged import run_capacity_experiment
    rows = run_capacity_experiment(config, capacities=[3] * 8, **options())
    assert [r['stage'] for r in rows] == ['contiguous', 'paged']
    for row in rows:
        assert row['experiment'] == 'clean_capacity'
        assert row['effective_budget_bytes'] == 6144 and row['budget_remainder_bytes'] == 356
        assert row['used_bytes'] == 0 and row['probe_admitted'] is None
        assert row['peak_memory_bytes'] is None and row['peak_memory_kind'] == 'unmeasured'
    assert rows[0]['live_requests'] == 5 and rows[0]['resident_bytes'] == rows[0]['reserved_bytes'] == 5760
    assert rows[0]['rounding_bytes'] == 0 and rows[0]['unused_logical_bytes'] == 5760
    assert rows[1]['live_requests'] == 4 and rows[1]['resident_bytes'] == rows[1]['reserved_bytes'] == 6144
    assert rows[1]['rounding_bytes'] == 1536 and rows[1]['unused_logical_bytes'] == 4608
    assert rows[0]['allocation_kind'] == 'real_contiguous_tensors' and rows[1]['allocation_kind'] == 'real_page_pool'


def test_fragmentation_trace_real_pages_vs_metadata(config):
    from benchmarks.bench_paged import run_fragmentation_experiment
    rows = run_fragmentation_experiment(config, **options())
    for row in rows:
        assert row['experiment'] == 'fragmentation_trace'
        assert row['free_slots_before_probe'] == 8 and row['largest_free_run_before_probe'] == 4
        trace = json.loads(row['allocation_trace'])
        assert [step['op'] for step in trace] == ['allocate'] * 4 + ['free'] * 2 + ['allocate']
        assert trace[-1]['capacity'] == 8 and trace[-1]['accepted'] == row['probe_admitted']
    assert rows[0]['probe_admitted'] is False and rows[0]['live_requests'] == 2
    assert rows[0]['reserved_bytes'] == 3072 and rows[0]['resident_bytes'] is None
    assert rows[0]['allocation_kind'] == 'metadata_first_fit_arena'
    assert rows[0]['free_slots_after_probe'] == 8 and rows[0]['largest_free_run_after_probe'] == 4
    assert rows[1]['probe_admitted'] is True and rows[1]['live_requests'] == 3
    assert rows[1]['reserved_bytes'] == 6144 and rows[1]['resident_bytes'] == 6144
    assert rows[1]['free_slots_after_probe'] == 0 and rows[1]['largest_free_run_after_probe'] == 0
    assert tuple(json.loads(rows[1]['allocation_trace'])[-1]['page_ids']) == (0, 2)


def test_first_fit_coalesces_and_failed_probe_preserves_state():
    from benchmarks.bench_paged import _Arena
    arena = _Arena(16)
    assert arena.allocate('A', 4) and arena.allocate('B', 4) and arena.allocate('C', 4) and arena.allocate('D', 4)
    arena.release('A')
    arena.release('B')
    assert arena.free_slots == 8 and arena.largest_free_run == 8
    assert not arena.allocate('E', 9)
    assert arena.free_slots == 8 and arena.largest_free_run == 8
    assert arena.allocate('E', 8) and arena.free_slots == 0


@pytest.mark.parametrize('changes', [
    {'capacities': []}, {'capacities': [True]}, {'capacities': [65]}, {'capacities': [0]},
    {'capacities': [1.5]}, {'budget_bytes': 0}, {'budget_bytes': True}, {'budget_bytes': 100},
    {'page_size': 0}, {'page_size': True}, {'device': torch.device('meta')}, {'dtype': torch.int64},
])
def test_bad_capacity_workload_before_allocation(config, monkeypatch, changes):
    from benchmarks.bench_paged import run_capacity_experiment
    def forbidden(*args, **kwargs):
        raise AssertionError('bad workload allocated tensors')
    monkeypatch.setattr(torch, 'empty', forbidden)
    with pytest.raises(ValueError):
        run_capacity_experiment(config, **({'capacities': [3]} | options() | changes))


@pytest.mark.parametrize('changes', [{'budget_bytes': 4608}, {'page_size': 33}])
def test_bad_fragment_trace_before_allocation(config, monkeypatch, changes):
    from benchmarks.bench_paged import run_fragmentation_experiment
    def forbidden(*args, **kwargs):
        raise AssertionError('bad trace allocated tensors')
    monkeypatch.setattr(torch, 'empty', forbidden)
    with pytest.raises(ValueError):
        run_fragmentation_experiment(config, **(options() | changes))


def test_capacity_exception_returns_prior_pages(config, monkeypatch):
    from benchmarks import bench_paged
    pools = []
    original_pool, original_cache = bench_paged.PagePool, bench_paged.PagedKVCache
    def observe(*args, **kwargs):
        pool = original_pool(*args, **kwargs)
        pools.append(pool)
        return pool
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('allocation fault')
        return original_cache(*args, **kwargs)
    monkeypatch.setattr(bench_paged, 'PagePool', observe)
    monkeypatch.setattr(bench_paged, 'PagedKVCache', fail_second)
    with pytest.raises(RuntimeError, match='allocation fault'):
        bench_paged.run_capacity_experiment(config, capacities=[3] * 8, **options())
    assert pools and all(pool.free_pages == pool.num_pages for pool in pools)


def test_csv_roundtrip_preserves_scope_and_blank_cells(config):
    from benchmarks.bench_paged import run_capacity_experiment, run_fragmentation_experiment
    rows = run_capacity_experiment(config, capacities=[3] * 8, **options()) + run_fragmentation_experiment(config, **options())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    assert '\r' not in buffer.getvalue()
    buffer.seek(0)
    parsed = list(csv.DictReader(buffer))
    assert len(parsed) == 4 and all(row['peak_memory_bytes'] == '' for row in parsed)
    assert parsed[2]['resident_bytes'] == '' and parsed[0]['probe_admitted'] == ''
    assert json.loads(parsed[0]['request_capacities']) == [3] * 8
    assert json.loads(parsed[0]['model_config'])['hidden_size'] == 24


@pytest.mark.parametrize('corrupt', [False, True])
def test_real_inference_gate_and_pool_release(config, monkeypatch, corrupt):
    from benchmarks import bench_paged
    torch.manual_seed(31)
    model = GPT2Model(config).eval()
    original_result = bench_paged.Scheduler.result
    original_pool = bench_paged.PagePool
    pools = []
    def observe(*args, **kwargs):
        pool = original_pool(*args, **kwargs)
        pools.append(pool)
        return pool
    monkeypatch.setattr(bench_paged, 'PagePool', observe)
    if corrupt:
        def wrong(self, name):
            result = original_result(self, name)
            if self.page_pool is not None:
                result[-1] = (result[-1] + 1) % 37
            return result
        monkeypatch.setattr(bench_paged.Scheduler, 'result', wrong)
    call = lambda: bench_paged.check_inference(model, [torch.tensor([1]), torch.tensor([2])], [5, 8],
        num_pages=4, page_size=4, max_batch_size=2, pad_token_id=0)
    if corrupt:
        with pytest.raises(RuntimeError, match='output'):
            call()
    else:
        call()
    assert pools and all(pool.free_pages == pool.num_pages for pool in pools)
