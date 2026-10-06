"""Mixed workload metrics count useful tokens and completion latency honestly."""
import csv
import io
import json

import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model
from engine.generate import generate


@pytest.fixture
def model():
    torch.manual_seed(2)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()


@pytest.mark.parametrize('continuous,stage,extra', [(False, 'static_cohorts', 2), (True, 'continuous', 0)])
def test_useful_tokens_latency_memory_and_csv(model, continuous, stage, extra):
    from benchmarks.bench_scheduler import run_scheduler_workload
    prompts = [torch.tensor([1]), torch.tensor([2, 3, 4]), torch.tensor([5])]
    row = run_scheduler_workload(model, prompts, [1, 3, 1], pad_token_id=0,
        max_batch_size=2, repetitions=3, continuous=continuous)
    assert row['stage'] == stage and row['useful_generated_tokens'] == 5
    assert row['extra_static_tokens'] == extra
    assert row['tokens_per_second'] * row['median_seconds'] == pytest.approx(5)
    assert row['completion_p95_ms'] >= row['completion_p50_ms'] >= 0
    assert row['peak_kv_bytes'] == 4608
    assert row['peak_memory_bytes'] is None and row['peak_memory_kind'] == 'unmeasured'
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row), lineterminator='\n')
    writer.writeheader()
    writer.writerow(row)
    buffer.seek(0)
    parsed = next(csv.DictReader(buffer))
    assert parsed['peak_memory_bytes'] == ''
    assert json.loads(parsed['prompt_lengths']) == [1, 3, 1]
    assert json.loads(parsed['output_budgets']) == [1, 3, 1]


def test_static_results_available_at_cohort_return(model):
    from benchmarks.bench_scheduler import _run_once
    prompts = [torch.tensor([1]), torch.tensor([2, 3, 4]), torch.tensor([5])]
    outputs, completions, elapsed, peak, process_peak = _run_once(model, prompts, [1, 3, 1],
        pad_token_id=0, max_batch_size=2, continuous=False)
    assert completions[0] == completions[1] and completions[2] >= completions[1]
    assert elapsed >= max(completions) and peak == 4608 and process_peak is None
    for actual, prompt, budget in zip(outputs, prompts, [1, 3, 1]):
        assert torch.equal(actual, generate(model, prompt[None], budget)[0])


@pytest.mark.parametrize('continuous', [False, True])
def test_one_request_one_token(model, continuous):
    from benchmarks.bench_scheduler import run_scheduler_workload
    row = run_scheduler_workload(model, [torch.tensor([1])], [1], pad_token_id=0,
        max_batch_size=1, repetitions=3, continuous=continuous)
    assert row['useful_generated_tokens'] == 1 and row['extra_static_tokens'] == 0
    assert row['completion_p50_ms'] >= 0 and row['completion_p95_ms'] >= row['completion_p50_ms']
    assert row['peak_kv_bytes'] == (384 if continuous else 768)


@pytest.mark.parametrize('changes', [
    {'prompts': [], 'budgets': []}, {'budgets': [1, 2]}, {'budgets': [0]}, {'budgets': [-1]},
    {'budgets': [True]}, {'budgets': [1.5]}, {'budgets': [16]}, {'repetitions': 2},
    {'max_batch_size': 0}, {'max_batch_size': True}, {'pad_token_id': 37},
    {'prompts': [torch.tensor([1.])]},
    {'prompts': [torch.ones(1, dtype=torch.long, device='meta')]},
    {'prompts': [torch.ones(15, dtype=torch.long), torch.tensor([1])], 'budgets': [1, 15]},
])
@pytest.mark.parametrize('continuous', [False, True])
def test_bad_workload_rejected_before_forward(model, monkeypatch, changes, continuous):
    from benchmarks.bench_scheduler import run_scheduler_workload
    def forbidden(*args, **kwargs):
        raise AssertionError('bad workload forwarded')
    monkeypatch.setattr(model, 'forward', forbidden)
    options = {'prompts': [torch.tensor([1])], 'budgets': [1], 'pad_token_id': 0,
               'max_batch_size': 2, 'repetitions': 3, 'continuous': continuous} | changes
    with pytest.raises(ValueError):
        run_scheduler_workload(model, **options)
