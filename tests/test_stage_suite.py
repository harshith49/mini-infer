"""Uniform useful-work metrics, stage parity, cleanup and atomic results."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from benchmarks import stage_suite as suite
from engine.config import ModelConfig
from engine.generate import generate
from engine.model import GPT2Model
from engine.quantize import quantize_model


@pytest.fixture
def model():
    torch.manual_seed(2)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()


@pytest.fixture
def workload():
    return suite.Workload((2, 5, 3), (1, 3, 2), page_size=2, num_pages=8)


@pytest.mark.parametrize('stage', suite.STAGES[:5])
def test_stage_parity_and_useful_work(model, workload, stage):
    prompts = suite.make_prompts(model.config, workload, torch.device('cpu'))
    expected = [generate(model, p[None], b, use_cache=True)[0].cpu()
                for p, b in zip(prompts, workload.output_budgets)]
    result = suite.run_stage_once(model, prompts, workload, stage)
    assert len(result.outputs) == 3
    assert all(torch.equal(a, b) for a, b in zip(result.outputs, expected))
    assert len(result.completions) == 3
    assert all(0 < t <= result.elapsed for t in result.completions)
    assert result.peak_memory_bytes is None
    row = suite.measure(lambda: suite.run_stage_once(model, prompts, workload, stage), expected, workload)
    assert row['useful_generated_tokens'] == 6
    assert row['completion_p95_seconds'] >= row['completion_p50_seconds'] > 0
    assert row['peak_memory_kind'] == 'unmeasured'
    assert suite.metadata(model, prompts, workload, stage, torch.device('cpu'))['extra_static_tokens'] == (2 if stage == 'static_batching' else 0)


def test_int8_matches_own_warmup(model, workload):
    quantize_model(model)
    prompts = suite.make_prompts(model.config, workload, torch.device('cpu'))
    expected = suite.run_stage_once(model, prompts, workload, 'int8').outputs
    assert suite.measure(lambda: suite.run_stage_once(model, prompts, workload, 'int8'), expected, workload)['useful_generated_tokens'] == 6


@pytest.mark.parametrize('field,value', [('repetitions', 2), ('seed', True), ('num_pages', 0), ('page_size', -1), ('max_batch_size', True), ('output_budgets', (1,)), ('prompt_lengths', (0, 2, 3))])
def test_invalid_workload(workload, field, value):
    with pytest.raises(ValueError):
        replace(workload, **{field: value})


def test_context_and_pool_validation(model, workload):
    for bad in [replace(workload, prompt_lengths=(15, 5, 3)), replace(workload, num_pages=1)]:
        with pytest.raises(ValueError):
            suite.make_prompts(model.config, bad, torch.device('cpu'))


def test_metrics_exact(workload):
    ids = [torch.arange(n) for n in (3, 8, 5)]
    elapsed = iter([9., 1., 2., 3.])
    def run():
        seconds = next(elapsed)
        return suite.RunResult(ids, [seconds/4, seconds/2, seconds], seconds, 10, None)
    row = suite.measure(run, ids, workload)
    assert row['median_seconds'] == 2
    assert row['tokens_per_second'] == 3
    assert row['peak_kv_bytes'] == 10
    from benchmarks.load_test import percentile
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert percentile([1, 2, 3, 4], .95) == pytest.approx(3.85)


def test_mismatch_rejected(workload):
    with pytest.raises(RuntimeError):
        suite.measure(lambda: suite.RunResult([torch.tensor([2])] * 3, [1.] * 3, 1., 0, None), [torch.tensor([1])] * 3, workload)


def test_atomic_csv(tmp_path, monkeypatch):
    dest = tmp_path / 'result.csv'
    dest.write_text('previous')
    with pytest.raises(ValueError):
        suite.write_rows(dest, [{'a': 1}, {'b': 2}])
    assert dest.read_text() == 'previous'
    def broken(*args):
        raise OSError('replace failure')
    monkeypatch.setattr(suite.os, 'replace', broken)
    with pytest.raises(OSError):
        suite.write_rows(dest, [{'a': 1}])
    assert dest.read_text() == 'previous'
    assert list(tmp_path.iterdir()) == [dest]


def test_scheduler_cleanup_on_fault(model, workload, monkeypatch):
    prompts = suite.make_prompts(model.config, workload, torch.device('cpu'))
    closed = []
    original = suite.Scheduler.close
    def close(self):
        original(self)
        closed.append(self.page_pool.owned_pages)
    monkeypatch.setattr(suite.Scheduler, 'close', close)
    def broken(*args, **kwargs):
        raise RuntimeError('forward failed')
    monkeypatch.setattr(model, 'forward', broken)
    with pytest.raises(RuntimeError, match='forward failed'):
        suite.run_stage_once(model, prompts, workload, 'paged_kv')
    assert closed == [0]


def test_suite_quality_and_order(model, workload, monkeypatch):
    from benchmarks import bench_quantization as quality
    monkeypatch.setattr(quality, 'load_quality_text', lambda *args: 'text')
    monkeypatch.setattr(quality, 'quality_input_ids', lambda *args, **kwargs: torch.tensor([[1,2,3]]))
    monkeypatch.setattr(quality, 'evaluate_perplexity', lambda *args, **kwargs: dict(scored_tokens=2, nll=2., perplexity=7.))
    rows = suite.run_suite(model, None, workload)
    assert [r['stage'] for r in rows] == list(suite.STAGES)
    assert all(r['useful_generated_tokens'] == 6 for r in rows)
    assert rows[-1]['relative_ppl_delta'] == 0
    assert len({tuple(r) for r in rows}) == 1
    with pytest.raises(ValueError):
        suite.run_suite(model, None, workload)


def test_quality_failure_does_not_publish(model, workload, monkeypatch):
    from benchmarks import bench_quantization as quality
    monkeypatch.setattr(quality, 'load_quality_text', lambda *args: 'text')
    monkeypatch.setattr(quality, 'quality_input_ids', lambda *args, **kwargs: torch.tensor([[1,2,3]]))
    values = iter([7., 8.])
    monkeypatch.setattr(quality, 'evaluate_perplexity', lambda *args, **kwargs: dict(scored_tokens=2, nll=2., perplexity=next(values)))
    with pytest.raises(ValueError, match='quality'):
        suite.run_suite(model, None, workload)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_peak_and_repeated_parity(model, workload):
    model = model.cuda()
    prompts = suite.make_prompts(model.config,workload,torch.device('cuda'))
    expected = [generate(model,p[None],b,use_cache=True)[0].cpu() for p,b in zip(prompts,workload.output_budgets)]
    row = suite.measure(lambda: suite.run_stage_once(model,prompts,workload,'paged_kv'),expected,workload)
    assert row['peak_memory_kind'] == 'cuda_allocated_including_model'
    assert row['peak_memory_bytes'] >= suite.model_storage_bytes(model)['model_bytes']


def test_scheduler_cleanup_success(model, workload, monkeypatch):
    prompts = suite.make_prompts(model.config,workload,torch.device('cpu'))
    closed = []
    original = suite.Scheduler.close
    def close(self):
        original(self)
        closed.append(self.page_pool.owned_pages)
    monkeypatch.setattr(suite.Scheduler,'close',close)
    suite.run_stage_once(model,prompts,workload,'paged_kv')
    assert closed == [0]


@pytest.mark.parametrize('problem', ['short', 'count', 'rank', 'actual_count', 'actual_rank', 'dtype'])
def test_measure_requires_workload_outputs(workload, problem):
    expected = [torch.arange(n) for n in (3, 8, 5)]
    if problem == 'short': expected = [torch.arange(n) for n in workload.prompt_lengths]
    elif problem == 'count': expected = expected[:2]
    elif problem == 'rank': expected = [x[None] for x in expected]
    elif problem == 'dtype': expected = [x.float() for x in expected]
    actual = expected
    if problem == 'actual_count': actual = expected[:2]
    elif problem == 'actual_rank': actual = [x[None] for x in expected]
    with pytest.raises(RuntimeError):
        suite.measure(lambda: suite.RunResult(actual, [1.] * len(actual), 1., 0, None), expected, workload)
