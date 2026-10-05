"""Measurements report scopes honestly; no wall-clock speed assertions."""
import csv
import io

import pytest
import torch

from benchmarks.bench_stages import run_workload
from engine.config import ModelConfig
from engine.model import GPT2Model


@pytest.fixture
def model():
    torch.manual_seed(2)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
                               num_layers=2, num_heads=4, intermediate_size=96)).eval()


@pytest.mark.parametrize('use_cache,stage,cache_bytes', [
    (False, 'naive', 0), (True, 'kv_cache', 2 * 2 * 1 * 4 * 5 * 6 * 4)])
def test_metrics_and_csv_scopes(model, use_cache, stage, cache_bytes):
    row = run_workload(model, torch.tensor([[1, 2]]), max_new_tokens=3,
                       repetitions=3, use_cache=use_cache)
    assert row['stage'] == stage
    assert row['device'] == 'cpu'
    assert row['generated_tokens'] == 3 and row['prompt_tokens'] == 2
    assert row['repetitions'] == 3
    assert row['tokens_per_second'] > 0 and row['median_seconds'] > 0
    assert row['ttft_seconds'] >= 0
    assert row['decode_p50_ms'] >= 0 and row['decode_p95_ms'] >= row['decode_p50_ms']
    assert row['cache_allocated_bytes'] == cache_bytes
    assert row['peak_memory_bytes'] is None
    assert row['peak_memory_kind'] == 'unmeasured'
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    buffer.seek(0)
    parsed = next(csv.DictReader(buffer))
    assert parsed['peak_memory_bytes'] == ''
    assert parsed['cache_allocated_bytes'] == str(cache_bytes)


@pytest.mark.parametrize('use_cache', [False, True])
def test_one_token_has_no_decode_latency(model, use_cache):
    row = run_workload(model, torch.tensor([[1]]), max_new_tokens=1,
                       repetitions=3, use_cache=use_cache)
    assert row['decode_p50_ms'] is None and row['decode_p95_ms'] is None


@pytest.mark.parametrize('tokens,reps', [(0, 3), (-1, 3), (17, 3), (3, 2), (3, 0)])
def test_invalid_workload_rejected(model, tokens, reps):
    with pytest.raises(ValueError):
        run_workload(model, torch.tensor([[1]]), max_new_tokens=tokens,
                     repetitions=reps, use_cache=True)
