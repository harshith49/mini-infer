"""Cache state, offset masking, and transactional prefix checks."""
import pytest
import torch

from engine.config import ModelConfig
from engine.kv_cache import SimpleKVCache
from engine.model import GPT2Model


@pytest.fixture
def model():
    torch.manual_seed(31)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
                               num_layers=2, num_heads=4, intermediate_size=96)).eval()


def cache_for(model, capacity=16, **kwargs):
    return SimpleKVCache(model.config, batch_size=kwargs.get('batch_size', 1),
        capacity=capacity, device=torch.device('cpu'), dtype=kwargs.get('dtype', torch.float32))


@torch.inference_mode()
def test_cache_byte_accounting_and_single_length_commit(model):
    cache = cache_for(model)
    assert cache.keys.shape == (2, 1, 4, 16, 6)
    assert cache.values.shape == cache.keys.shape
    assert cache.allocated_bytes == 2 * 2 * 1 * 4 * 16 * 6 * 4
    assert cache.used_bytes == 0
    model(torch.tensor([[1, 2, 3]]), cache=cache)
    assert cache.length == 3
    assert cache.used_bytes == 2 * 2 * 1 * 4 * 3 * 6 * 4
    assert cache.allocated_bytes == 2 * 2 * 1 * 4 * 16 * 6 * 4


@pytest.mark.parametrize('chunks', [[1, 1, 1, 1], [2, 2], [1, 3], [4]])
@torch.inference_mode()
def test_cached_chunk_logits_match_uncached(model, chunks):
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids)
    cache = cache_for(model)
    actual = []
    offset = 0
    for size in chunks:
        actual.append(model(ids[:, offset:offset + size], cache=cache))
        offset += size
        assert cache.length == offset
    torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_requests_are_independent_and_exact_capacity_works(model):
    first, second = cache_for(model, 4), cache_for(model, 4)
    model(torch.tensor([[1, 2]]), cache=first)
    assert second.length == 0
    torch.testing.assert_close(model(torch.tensor([[3, 4]]), cache=second),
                                model(torch.tensor([[3, 4]])), atol=1e-4, rtol=1e-4)
    logits = model(torch.tensor([[3, 4]]), cache=first)
    torch.testing.assert_close(logits, model(torch.tensor([[1, 2, 3, 4]]))[:, 2:],
                               atol=1e-4, rtol=1e-4)
    assert first.length == 4
    prefix = first.keys.clone()
    with pytest.raises(ValueError, match='capacity'):
        model(torch.tensor([[5]]), cache=first)
    assert first.length == 4
    torch.testing.assert_close(first.keys, prefix, atol=0, rtol=0)


@pytest.mark.parametrize('changes', [{'batch_size': 2}, {'dtype': torch.float64}])
@torch.inference_mode()
def test_incompatible_cache_rejected_before_writes(model, changes):
    cache = cache_for(model, **changes)
    cache.keys.fill_(23)
    cache.values.fill_(29)
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 2]]), cache=cache)
    assert cache.length == 0
    assert torch.all(cache.keys == 23) and torch.all(cache.values == 29)


@torch.inference_mode()
def test_wrong_model_dimensions_rejected(model):
    other = ModelConfig(vocab_size=37, max_positions=16, hidden_size=32,
                        num_layers=2, num_heads=4, intermediate_size=96)
    cache = SimpleKVCache(other, batch_size=1, capacity=16,
                          device=torch.device('cpu'), dtype=torch.float32)
    with pytest.raises(ValueError):
        model(torch.tensor([[1]]), cache=cache)
    assert cache.length == 0


@torch.inference_mode()
def test_failed_late_layer_leaves_prefix_reusable(model, monkeypatch):
    cache = cache_for(model)
    model(torch.tensor([[1, 2]]), cache=cache)
    saved_keys = cache.keys[:, :, :, :2].clone()
    saved_values = cache.values[:, :, :, :2].clone()
    original = model.blocks[1].forward

    def fail(*args, **kwargs):
        raise RuntimeError('injected late layer fault')

    monkeypatch.setattr(model.blocks[1], 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        model(torch.tensor([[3, 4]]), cache=cache)
    assert cache.length == 2
    torch.testing.assert_close(cache.keys[:, :, :, :2], saved_keys, atol=0, rtol=0)
    torch.testing.assert_close(cache.values[:, :, :, :2], saved_values, atol=0, rtol=0)
    monkeypatch.setattr(model.blocks[1], 'forward', original)
    torch.testing.assert_close(model(torch.tensor([[3, 4]]), cache=cache),
        model(torch.tensor([[1, 2, 3, 4]]))[:, 2:], atol=1e-4, rtol=1e-4)
    assert cache.length == 4


@pytest.mark.parametrize('capacity,batch', [(0, 1), (17, 1), (4, 0), (-1, 1)])
def test_invalid_cache_allocation(model, capacity, batch):
    with pytest.raises(ValueError):
        SimpleKVCache(model.config, batch_size=batch, capacity=capacity,
                      device=torch.device('cpu'), dtype=torch.float32)
