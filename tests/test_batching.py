"""Padding, logical positions, and fixed-batch generation boundaries."""
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


def cache_for(model, capacity=16):
    return SimpleKVCache(model.config, batch_size=2, capacity=capacity,
                         device=torch.device('cpu'), dtype=torch.float32)


@torch.inference_mode()
@pytest.mark.parametrize('mask_dtype', [torch.bool, torch.long, torch.int32])
def test_padding_logits_positions_and_invariance(model, mask_dtype):
    ids = torch.tensor([[0, 0, 1], [2, 3, 4]])
    mask = torch.tensor([[0, 0, 1], [1, 1, 1]], dtype=mask_dtype)
    actual = model(ids, attention_mask=mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0, 2:], model(ids[0:1, 2:])[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual[1], model(ids[1:])[0], atol=1e-4, rtol=1e-4)
    explicit = model(ids, attention_mask=mask, position_ids=torch.tensor([[0, 0, 0], [0, 1, 2]]))
    torch.testing.assert_close(actual, explicit, atol=0, rtol=0)
    ids[0, :2] = torch.tensor([17, 23])
    changed = model(ids, attention_mask=mask)
    torch.testing.assert_close(changed[0, 2:], actual[0, 2:], atol=0, rtol=0)


@torch.inference_mode()
def test_padded_cached_chunk_exact_capacity(model):
    cache = cache_for(model, 5)
    model(torch.tensor([[0, 0, 1], [2, 3, 4]]), cache=cache,
          attention_mask=torch.tensor([[0, 0, 1], [1, 1, 1]]))
    actual = model(torch.tensor([[5, 6], [7, 8]]), cache=cache,
          attention_mask=torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))
    for row, ids in enumerate([[1, 5, 6], [2, 3, 4, 7, 8]]):
        torch.testing.assert_close(actual[row], model(torch.tensor([ids]))[0, -2:], atol=1e-4, rtol=1e-4)
    assert cache.length == 5 and cache.requires_attention_mask
    assert cache.used_bytes == cache.allocated_bytes == 2 * 2 * 2 * 4 * 5 * 6 * 4
    with pytest.raises(ValueError, match='capacity'):
        model(torch.tensor([[1], [2]]), cache=cache, attention_mask=torch.ones(2, 6, dtype=torch.bool))
    assert cache.length == 5


@torch.inference_mode()
def test_explicit_positions_without_mask(model):
    ids = torch.tensor([[1, 2], [3, 4]])
    positions = torch.tensor([[3, 4], [5, 6]])
    expected = model(ids, position_ids=positions)
    cache = cache_for(model, 2)
    chunks = [model(ids[:, i:i+1], cache=cache, position_ids=positions[:, i:i+1]) for i in range(2)]
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=1e-4, rtol=1e-4)
    assert not cache.requires_attention_mask


BAD_INPUTS = [
    {'attention_mask': torch.ones(2)},
    {'attention_mask': torch.ones(2, 2, dtype=torch.bool)},
    {'attention_mask': torch.ones(1, 3, dtype=torch.bool)},
    {'attention_mask': torch.ones(2, 3)},
    {'attention_mask': torch.tensor([[1, 2, 1], [1, 1, 1]])},
    {'attention_mask': torch.tensor([[0, 0, 0], [1, 1, 1]])},
    {'attention_mask': torch.ones(2, 3, dtype=torch.bool, device='meta')},
    {'position_ids': torch.tensor([0, 1])},
    {'position_ids': torch.zeros(1, 1, dtype=torch.long)},
    {'position_ids': torch.zeros(2, 1)},
    {'position_ids': torch.full((2, 1), -1)},
    {'position_ids': torch.full((2, 1), 16)},
    {'position_ids': torch.zeros(2, 1, dtype=torch.long, device='meta')},
]


@pytest.mark.parametrize('kwargs', BAD_INPUTS)
@torch.inference_mode()
def test_invalid_masks_positions_preserve_cache(model, kwargs):
    cache = cache_for(model)
    model(torch.tensor([[1, 2], [3, 4]]), cache=cache)
    # Initialize unused storage before equality comparisons (torch.empty may contain NaN).
    cache.keys[:, :, :, 2:] = 23
    cache.values[:, :, :, 2:] = 29
    keys, values = cache.keys.clone(), cache.values.clone()
    with pytest.raises(ValueError):
        model(torch.tensor([[5], [6]]), cache=cache, **kwargs)
    assert cache.length == 2 and not cache.requires_attention_mask
    torch.testing.assert_close(cache.keys, keys, atol=0, rtol=0)
    torch.testing.assert_close(cache.values, values, atol=0, rtol=0)


@torch.inference_mode()
def test_missing_mask_and_failed_projection_preserve_metadata(model, monkeypatch):
    cache = cache_for(model)
    prefix = torch.tensor([[1, 2], [3, 4]])
    model(prefix, cache=cache)
    keys, values = cache.keys[:, :, :, :2].clone(), cache.values[:, :, :, :2].clone()
    mask = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]])
    chunk = torch.tensor([[0, 5], [6, 7]])
    original = model.lm_head.forward
    def fail(*args, **kwargs):
        raise RuntimeError('injected projection fault')
    monkeypatch.setattr(model.lm_head, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        model(chunk, cache=cache, attention_mask=mask)
    assert cache.length == 2 and not cache.requires_attention_mask
    torch.testing.assert_close(cache.keys[:, :, :, :2], keys, atol=0, rtol=0)
    torch.testing.assert_close(cache.values[:, :, :, :2], values, atol=0, rtol=0)
    monkeypatch.setattr(model.lm_head, 'forward', original)
    actual = model(chunk, cache=cache, attention_mask=mask)
    expected = model(torch.cat((prefix, chunk), dim=1), attention_mask=mask)[:, -2:]
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert cache.length == 4 and cache.requires_attention_mask
    keys, values = cache.keys.clone(), cache.values.clone()
    with pytest.raises(ValueError, match='attention_mask'):
        model(torch.tensor([[8], [9]]), cache=cache)
    assert cache.length == 4 and cache.requires_attention_mask
    torch.testing.assert_close(cache.keys, keys, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(cache.values, values, atol=0, rtol=0, equal_nan=True)
