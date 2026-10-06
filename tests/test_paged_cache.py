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
    from engine.kv_cache import PagePool
    pool = PagePool(config, num_pages=4, device=torch.device('cpu'), dtype=torch.float32)
    assert pool.page_size == 16
    pool.allocate(3)
    with pytest.raises(MemoryError):
        pool.allocate(2)
    assert pool.free_page_ids == (3,) and pool.owned_pages == 3


def make_cache(pool, capacity):
    from engine.kv_cache import PagedKVCache
    return PagedKVCache(pool, capacity=capacity)


def test_paged_cross_boundary_mapping_and_tail_exclusion(config):
    pool = make_pool(config, num_pages=6)
    pool.allocate(6)
    pool.release((0, 2, 4))
    pool.keys.fill_(float('nan'))
    pool.values.fill_(float('nan'))
    cache = make_cache(pool, 9)
    assert cache.page_ids == (0, 2, 4)
    expected = torch.arange(9 * 24, dtype=torch.float32).reshape(1, 4, 9, 6)
    for layer in range(2):
        cache.store(layer, 0, expected[:, :, :3], expected[:, :, :3] + 100)
        cache.store(layer, 3, expected[:, :, 3:8], expected[:, :, 3:8] + 100)
        cache.length = 8
        keys, values = cache.write(layer, expected[:, :, 8:], expected[:, :, 8:] + 100)
        assert keys.is_contiguous() and values.is_contiguous()
        assert cache.length == 8
        torch.testing.assert_close(keys, expected, atol=0, rtol=0)
        torch.testing.assert_close(values, expected + 100, atol=0, rtol=0)
        assert cache.read_prefix(layer)[0].shape[-2] == 8
    cache.close()
    pool.release((1, 3, 5))
    assert pool.free_pages == 6


def test_paged_byte_counts_and_explicit_close(config):
    pool = make_pool(config)
    cache = make_cache(pool, 9)
    cache.length = 5
    assert cache.allocated_bytes == 4608 and cache.used_bytes == 1920
    assert cache.rounding_bytes == 1152 and cache.unused_bytes == 1536
    cache.close()
    cache.close()
    assert cache.page_ids == () and cache.length == 0
    assert cache.allocated_bytes == cache.used_bytes == cache.rounding_bytes == cache.unused_bytes == 0
    assert pool.free_pages == 4 and pool.allocated_bytes == 6144
    chunk = torch.ones(1, 4, 1, 6)
    for operation in [lambda: cache.read_prefix(0), lambda: cache.store(0, 0, chunk, chunk),
                      lambda: cache.write(0, chunk, chunk),
                      lambda: cache.validate(config, torch.tensor([[1]]), device=torch.device('cpu'), dtype=torch.float32)]:
        with pytest.raises(ValueError, match='closed'):
            operation()


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5, 65])
def test_bad_paged_capacity_preserves_pool(config, capacity):
    pool = make_pool(config)
    with pytest.raises(ValueError):
        make_cache(pool, capacity)
    assert pool.free_pages == 4


def test_paged_reservation_exhaustion_is_atomic(config):
    pool = make_pool(config)
    a = make_cache(pool, 9)
    with pytest.raises(MemoryError):
        make_cache(pool, 5)
    assert pool.free_page_ids == (3,) and a.page_ids == (0, 1, 2)
    a.close()
    assert pool.free_pages == 4


@pytest.mark.parametrize('changes', [
    {'layer': -1}, {'layer': 2}, {'offset': -1}, {'offset': 9}, {'offset': True},
    {'key': torch.ones(4, 1, 6)}, {'key': torch.ones(2, 4, 1, 6)},
    {'key': torch.ones(1, 4, 0, 6)}, {'key': torch.ones(1, 4, 1, 6, dtype=torch.float64)},
    {'value': torch.ones(1, 4, 1, 5)}, {'key': torch.ones(1, 4, 1, 6, device='meta')},
])
def test_paged_invalid_store_before_mutation(config, changes):
    pool = make_pool(config)
    pool.keys.zero_()
    pool.values.zero_()
    cache = make_cache(pool, 9)
    args = {'layer': 0, 'offset': 0, 'key': torch.ones(1, 4, 1, 6), 'value': torch.ones(1, 4, 1, 6)} | changes
    with pytest.raises(ValueError):
        cache.store(args['layer'], args['offset'], args['key'], args['value'])
    assert not pool.keys.any() and not pool.values.any() and cache.length == 0
    cache.close()


@pytest.mark.parametrize('length', [-1, 10, True, 1.5])
def test_bad_paged_read_length(config, length):
    cache = make_cache(make_pool(config), 9)
    with pytest.raises(ValueError):
        cache.read_prefix(0, length=length)
    cache.close()


@pytest.mark.parametrize('paged', [False, True])
def test_logical_operations_shared_by_both_caches(config, paged):
    from engine.kv_cache import SimpleKVCache
    cache = make_cache(make_pool(config), 9) if paged else SimpleKVCache(config,
        batch_size=1, capacity=9, device=torch.device('cpu'), dtype=torch.float32)
    chunk = torch.randn(1, 4, 3, 6)
    cache.store(0, 0, chunk, chunk + 1)
    cache.length = 3
    keys, values = cache.read_prefix(0)
    torch.testing.assert_close(keys, chunk, atol=0, rtol=0)
    torch.testing.assert_close(values, chunk + 1, atol=0, rtol=0)
    if paged:
        assert keys.numel() * keys.element_size() * 2 == 576
        cache.close()
    else:
        assert keys.untyped_storage().data_ptr() == cache.keys.untyped_storage().data_ptr()


@pytest.mark.parametrize('chunks', [[3, 2, 4], [4, 1, 3], [8, 8], [32, 32]])
@torch.inference_mode()
def test_direct_paged_logits_match_contiguous(config, chunks):
    from engine.kv_cache import SimpleKVCache
    from engine.model import GPT2Model
    torch.manual_seed(31)
    model = GPT2Model(config).eval()
    pool = make_pool(config, num_pages=16)
    pool.keys.fill_(float('nan'))
    pool.values.fill_(float('nan'))
    paged = make_cache(pool, sum(chunks))
    contiguous = SimpleKVCache(config, batch_size=1, capacity=sum(chunks),
                               device=torch.device('cpu'), dtype=torch.float32)
    try:
        for size in chunks:
            ids = torch.randint(37, (1, size))
            torch.testing.assert_close(model(ids, cache=paged), model(ids, cache=contiguous), atol=1e-4, rtol=1e-4)
            assert paged.length == contiguous.length
    finally:
        paged.close()
    assert pool.free_pages == 16


@pytest.mark.parametrize('masked', [False, True])
@torch.inference_mode()
def test_paged_projection_failure_preserves_prefix_and_retry(config, masked):
    from engine.kv_cache import SimpleKVCache
    from engine.model import GPT2Model
    torch.manual_seed(31)
    model = GPT2Model(config).eval()
    pool = make_pool(config)
    paged = make_cache(pool, 9)
    contiguous = SimpleKVCache(config, batch_size=1, capacity=9, device=torch.device('cpu'), dtype=torch.float32)
    mask = torch.tensor([[False, True, True]]) if masked else None
    for cache in [paged, contiguous]:
        model(torch.tensor([[0, 1, 2]]), cache=cache, attention_mask=mask)
    before = [tuple(t.clone() for t in paged.read_prefix(i)) for i in range(2)]
    def fail(*args):
        raise RuntimeError('projection fault')
    hook = model.lm_head.register_forward_hook(fail)
    next_mask = torch.tensor([[False, True, True, True, True]]) if masked else None
    with pytest.raises(RuntimeError, match='projection fault'):
        model(torch.tensor([[3, 4]]), cache=paged, attention_mask=next_mask)
    hook.remove()
    assert paged.length == 3 and paged.requires_attention_mask == masked and pool.owned_pages == 3
    for layer in range(2):
        for actual, expected in zip(paged.read_prefix(layer), before[layer]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(model(torch.tensor([[3, 4]]), cache=paged, attention_mask=next_mask),
        model(torch.tensor([[3, 4]]), cache=contiguous, attention_mask=next_mask), atol=1e-4, rtol=1e-4)
    paged.close()
    assert pool.free_pages == 4


@pytest.mark.parametrize('changes', [
    {'input_ids': torch.tensor([[1], [2]])}, {'device': torch.device('meta')},
    {'dtype': torch.float64},
    {'config': ModelConfig(vocab_size=38, max_positions=64, hidden_size=24, num_layers=2, num_heads=4, intermediate_size=96)},
])
def test_paged_validate_rejects_incompatible_model(config, changes):
    pool = make_pool(config)
    cache = make_cache(pool, 9)
    options = {'config': config, 'input_ids': torch.tensor([[1]]),
               'device': torch.device('cpu'), 'dtype': torch.float32} | changes
    with pytest.raises(ValueError):
        cache.validate(**options)
    assert cache.length == 0 and pool.owned_pages == 3
    cache.close()
