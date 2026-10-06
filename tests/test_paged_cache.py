"""Atomic shared pages and logical request cache behavior."""
import pytest
import torch

from engine.config import ModelConfig


@pytest.fixture
def config():
    return ModelConfig(vocab_size=37, max_positions=64, hidden_size=24,
                       num_layers=2, num_heads=4, intermediate_size=96)


def make_pool(config, **options):
    from engine.kv_cache import PagePool
    return PagePool(config, **({'num_pages': 4, 'page_size': 4,
        'device': torch.device('cpu'), 'dtype': torch.float32} | options))


def test_pool_nonconsecutive_reuse_and_atomic_exhaustion(config):
    pool = make_pool(config)
    assert pool.keys.shape == (2, 4, 4, 4, 6)
    assert pool.values.shape == pool.keys.shape
    assert pool.allocate(2) == (0, 1)
    assert pool.allocate(2) == (2, 3)
    pool.release((0, 2))
    assert pool.free_page_ids == (0, 2) and pool.free_pages == 2 and pool.owned_pages == 2
    assert pool.allocate(2) == (0, 2)
    before = pool.free_page_ids
    with pytest.raises(MemoryError, match='page'):
        pool.allocate(1)
    assert pool.free_page_ids == before and pool.owned_pages == 4
    pool.release((0, 1, 2, 3))
    assert pool.free_pages == 4 and pool.owned_pages == 0
    assert pool.page_bytes == 1536 and pool.allocated_bytes == 6144
    assert pool.allocated_bytes == sum(t.numel() * t.element_size() for t in (pool.keys, pool.values))


@pytest.mark.parametrize('options', [
    {'num_pages': 0}, {'num_pages': -1}, {'num_pages': True}, {'num_pages': 1.5},
    {'page_size': 0}, {'page_size': -1}, {'page_size': True}, {'page_size': 1.5},
    {'dtype': torch.int64}, {'dtype': 'float32'},
])
def test_pool_bad_construction(config, options):
    with pytest.raises(ValueError):
        make_pool(config, **options)


@pytest.mark.parametrize('count', [0, -1, True, 1.5, '1', None])
def test_bad_allocation_preserves_free_ids(config, count):
    pool = make_pool(config)
    with pytest.raises(ValueError):
        pool.allocate(count)
    assert pool.free_page_ids == (0, 1, 2, 3)


@pytest.mark.parametrize('ids', [(0, 0), (0, 4), (0, -1), (0, 2), (0, True), (0, 1.5)])
def test_bad_release_is_atomic(config, ids):
    pool = make_pool(config)
    pool.allocate(2)
    before = pool.free_page_ids
    with pytest.raises(ValueError):
        pool.release(ids)
    assert pool.free_page_ids == before and pool.owned_pages == 2
    pool.release((0, 1))
    with pytest.raises(ValueError):
        pool.release((0,))
    assert pool.free_pages == 4 and pool.allocated_bytes == 6144


def test_default_page_size_and_partial_exhaustion(config):
    pool = make_pool(config, page_size=16)
    assert pool.page_size == 16
    pool.allocate(3)
    with pytest.raises(MemoryError):
        pool.allocate(2)
    assert pool.free_page_ids == (3,) and pool.owned_pages == 3
