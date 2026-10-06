"""Page-pressure FIFO admission, phase recovery, and real memory scopes."""
import json
import sys
import weakref

import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model
from engine.kv_cache import PagePool, PagedKVCache
from engine.generate import generate
from engine.sampler import SamplingParams


@pytest.fixture
def model():
    torch.manual_seed(31)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()


def scheduler_for(model, *, pages=4, size=4, limit=3):
    from engine.scheduler import Scheduler
    pool = PagePool(model.config, num_pages=pages, page_size=size,
        device=model.token_embedding.weight.device, dtype=model.token_embedding.weight.dtype)
    return Scheduler(model, max_batch_size=limit, pad_token_id=0, page_pool=pool), pool


def drain(scheduler):
    events = []
    for _ in range(100):
        if scheduler.idle:
            return events
        events.extend(scheduler.step())
    pytest.fail('finite requests stalled under page pressure')


def test_page_pressure_fifo_without_small_waiter_bypass(model):
    s, pool = scheduler_for(model)
    jobs = [('A', [1], 5), ('B', [2], 8), ('C', [3], 1)]
    for name, prompt, budget in jobs:
        s.submit(name, torch.tensor(prompt), budget)
    assert pool.free_pages == 4
    events = s.step()
    assert [e.request_id for e in events] == ['A'] and pool.free_pages == 2
    for _ in range(4):
        assert [e.request_id for e in s.step()] == ['A']
    assert pool.free_pages == 4
    assert [e.request_id for e in s.step()] == ['B', 'C']
    assert pool.free_pages == 1
    drain(s)
    for name, prompt, budget in jobs:
        assert torch.equal(s.result(name), generate(model, torch.tensor([prompt]), budget, use_cache=True)[0])
    assert s.idle and pool.free_pages == 4 and pool.owned_pages == 0
    assert s.cache_allocated_bytes == 0 and s.pool_resident_bytes == 6144


def test_dynamic_page_waiter_is_not_reordered(model):
    s, pool = scheduler_for(model)
    s.submit('A', torch.tensor([1]), 5)
    s.step()
    s.submit('B', torch.tensor([2]), 8)
    s.step()
    s.submit('C', torch.tensor([3]), 1)
    events = drain(s)
    seen = list(dict.fromkeys(e.request_id for e in events if e.request_id != 'A'))
    assert seen == ['B', 'C'] and pool.free_pages == 4


def test_impossible_request_rejects_before_mutation(model):
    s, pool = scheduler_for(model, pages=1)
    with pytest.raises(ValueError, match='pool'):
        s.submit('A', torch.tensor([1]), 5)
    assert s.idle and not s._requests and pool.free_pages == 1
    s.submit('A', torch.tensor([1]), 1)
    drain(s)
    assert pool.free_pages == 1


def test_zero_outputs_need_no_pages_even_with_long_prompt(model, monkeypatch):
    s, pool = scheduler_for(model, pages=1)
    def forbidden(*args, **kwargs):
        raise AssertionError('zero work allocated or forwarded')
    monkeypatch.setattr(pool, 'allocate', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    prompt = torch.ones(8, dtype=torch.long)
    s.submit('zero', prompt, 0)
    event, = s.step()
    assert event.token_id is None and event.finish_reason == 'length'
    assert torch.equal(s.result('zero'), prompt)
    assert pool.free_pages == 1 and s.peak_kv_bytes == 1536 and s.cache_allocated_bytes == 0


@pytest.mark.parametrize('stop', [False, True])
def test_first_token_completion_releases_reservation(model, stop):
    s, pool = scheduler_for(model, pages=2)
    token = generate(model, torch.tensor([[1]]), 1)[0, -1].item()
    s.submit('A', torch.tensor([1]), 5 if stop else 1, stop_token_ids=(token, 36) if stop else ())
    event, = s.step()
    assert event.finish_reason == ('stop' if stop else 'length')
    assert pool.free_pages == 2 and s._requests['A'].cache is None


@pytest.mark.parametrize('settings', [SamplingParams(temperature=10, seed=7),
    SamplingParams(temperature=5, top_k=8, top_p=.9, seed=7)])
def test_paged_sampling_matches_contiguous_with_different_admission(model, settings):
    from engine.scheduler import Scheduler
    before = torch.get_rng_state().clone()
    outputs = []
    for paged in [False, True]:
        s, pool = scheduler_for(model, limit=2) if paged else (Scheduler(model, max_batch_size=2, pad_token_id=0), None)
        for name, prompt, budget in [('A', [1], 5), ('B', [2], 8)]:
            s.submit(name, torch.tensor(prompt), budget, sampling=settings)
        s.step()
        s.submit('C', torch.tensor([3]), 1, sampling=SamplingParams(seed=22))
        drain(s)
        outputs.append([s.result(name) for name in ['A', 'B', 'C']])
        if pool is not None:
            assert pool.free_pages == 4
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize('failure', ['second_reservation', 'projection'])
def test_staged_admission_failure_returns_pages_and_preserves_state(model, monkeypatch, failure):
    s, pool = scheduler_for(model)
    for name in ['A', 'B']:
        s.submit(name, torch.tensor([1]), 3, sampling=SamplingParams(temperature=2, seed=7))
    states = {name: r.generator.get_state().clone() for name, r in s._requests.items()}
    if failure == 'second_reservation':
        original = pool.allocate
        calls = 0
        def fail(count):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError('reservation fault')
            return original(count)
        monkeypatch.setattr(pool, 'allocate', fail)
        error = MemoryError
        cleanup = lambda: monkeypatch.setattr(pool, 'allocate', original)
    else:
        def fail(*args):
            raise RuntimeError('projection fault')
        hook = model.lm_head.register_forward_hook(fail)
        error = RuntimeError
        cleanup = hook.remove
    with pytest.raises(error, match='fault'):
        s.step()
    cleanup()
    assert pool.free_pages == 4 and list(s._waiting) == ['A', 'B'] and not s._running
    for name, r in s._requests.items():
        assert torch.equal(r.output, torch.tensor([1])) and r.cache is None
        assert torch.equal(r.generator.get_state(), states[name])
    assert [e.request_id for e in s.step()] == ['A', 'B']
    drain(s)
    assert pool.free_pages == 4


def test_admission_events_survive_old_decode_failure_once(model):
    s, pool = scheduler_for(model, pages=8, limit=2)
    s.submit('A', torch.tensor([1]), 5)
    s.step()
    r = s._requests['A']
    prefix = [tuple(t.clone() for t in r.cache.read_prefix(layer)) for layer in range(2)]
    rng = r.generator.get_state().clone()
    s.submit('B', torch.tensor([2]), 3)
    calls = 0
    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('decode fault')
    hook = model.lm_head.register_forward_hook(fail_second)
    with pytest.raises(RuntimeError, match='decode fault'):
        s.step()
    hook.remove()
    assert r.cache.length == 1 and len(r.output) == 2 and pool.owned_pages == 3
    assert torch.equal(r.generator.get_state(), rng)
    for layer in range(2):
        for actual, expected in zip(r.cache.read_prefix(layer), prefix[layer]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    retry = s.step()
    assert [e.request_id for e in retry] == ['B', 'A', 'B']
    events = retry + drain(s)
    assert sum(e.request_id == 'B' for e in events) == 3
    for name, token, budget in [('A', 1, 5), ('B', 2, 3)]:
        assert torch.equal(s.result(name), generate(model, torch.tensor([[token]]), budget, use_cache=True)[0])
    assert pool.free_pages == 8


def test_resident_and_gather_peak_bytes_and_lifetime(model, monkeypatch):
    s, pool = scheduler_for(model)
    previous = []
    original = PagedKVCache.read_prefix
    def observe(self, *args, **kwargs):
        assert all(ref() is None for ref in previous)
        pair = original(self, *args, **kwargs)
        previous[:] = [weakref.ref(t) for t in pair]
        return pair
    monkeypatch.setattr(PagedKVCache, 'read_prefix', observe)
    def before_forward(*args):
        assert all(ref() is None for ref in previous)
    hook = model.register_forward_pre_hook(before_forward)
    s.submit('A', torch.tensor([1]), 5)
    s.step()
    assert s.pool_resident_bytes == 6144 and s.cache_allocated_bytes == 3072 and s.peak_kv_bytes == 6528
    s.step()
    assert s.peak_kv_bytes == 7104
    drain(s)
    hook.remove()
    assert s.cache_allocated_bytes == 0 and s.pool_resident_bytes == 6144 and pool.free_pages == 4


def test_stale_reused_pages_and_prompt_result_isolation(model):
    s, pool = scheduler_for(model, pages=2)
    pool.keys.fill_(float('nan'))
    pool.values.fill_(float('nan'))
    prompt = torch.tensor([1])
    s.submit('A', prompt, 5)
    prompt[0] = 36
    drain(s)
    saved = s.result('A')
    s.result('A')[0] = 35
    assert torch.equal(s.result('A'), saved)
    s.submit('B', torch.tensor([2]), 5)
    drain(s)
    assert torch.equal(s.result('B'), generate(model, torch.tensor([[2]]), 5, use_cache=True)[0])
    assert pool.free_pages == 2


@pytest.mark.parametrize('mismatch', ['config', 'dtype', 'occupied'])
def test_pool_attachment_rejects_bad_pool(model, mismatch):
    from engine.scheduler import Scheduler
    config = model.config if mismatch != 'config' else ModelConfig(vocab_size=38, max_positions=16,
        hidden_size=24, num_layers=2, num_heads=4, intermediate_size=96)
    pool = PagePool(config, num_pages=4, page_size=4, device=torch.device('cpu'),
        dtype=torch.float64 if mismatch == 'dtype' else torch.float32)
    if mismatch == 'occupied':
        pool.allocate(1)
    before = pool.free_page_ids
    with pytest.raises(ValueError):
        Scheduler(model, max_batch_size=2, pad_token_id=0, page_pool=pool)
    assert pool.free_page_ids == before


@pytest.mark.parametrize('flags', [['--num-pages', '0'], ['--page-size', '0']])
def test_cli_bad_pool_settings_before_loading(model, monkeypatch, flags):
    from engine import scheduler
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid flags loaded weights')
    monkeypatch.setattr(scheduler, 'load_model', forbidden)
    monkeypatch.setattr(sys, 'argv', ['scheduler', '--prompt', 'Hi', '--cache-backend', 'paged'] + flags)
    with pytest.raises(SystemExit) as error:
        scheduler.main()
    assert error.value.code == 2


class Tokenizer:
    eos_token_id = 0
    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([[ord(char) % 36 + 1 for char in text]])}
    def decode(self, ids, **kwargs):
        return ' '.join(str(i) for i in ids if i != 0)


@pytest.mark.parametrize('backend', ['contiguous', 'paged'])
def test_cli_budgets_order_and_literal_prompts(model, monkeypatch, capsys, backend):
    from engine import scheduler
    tokenizer = Tokenizer()
    texts = ['', 'Café\n', '<|endoftext|>']
    budgets = [0, 3, 1]
    monkeypatch.setattr(scheduler, 'load_model', lambda *args: (model, tokenizer))
    if backend == 'contiguous':
        def forbidden(*args, **kwargs):
            raise AssertionError('contiguous CLI allocated a page pool')
        monkeypatch.setattr(scheduler, 'PagePool', forbidden, raising=False)
    argv = ['scheduler', '--cache-backend', backend, '--num-pages', '4', '--page-size', '4',
            '--max-new-tokens', '0', '3', '1']
    for text in texts:
        argv += ['--prompt', text]
    monkeypatch.setattr(sys, 'argv', argv)
    scheduler.main()
    actual = json.loads(capsys.readouterr().out)
    expected = []
    for text, budget in zip(texts, budgets):
        prompt = tokenizer(text)['input_ids'] if text else torch.tensor([[0]])
        output = generate(model, prompt, budget, eos_token_id=0, use_cache=True)
        expected.append(text + tokenizer.decode(output[0, prompt.shape[1]:].tolist()))
    assert actual == expected


def test_cli_paged_impossible_request_errors_cleanly(model, monkeypatch, capsys):
    from engine import scheduler
    monkeypatch.setattr(scheduler, 'load_model', lambda *args: (model, Tokenizer()))
    monkeypatch.setattr(sys, 'argv', ['scheduler', '--cache-backend', 'paged', '--num-pages', '1',
                                     '--page-size', '4', '--prompt', 'Hi', '--max-new-tokens', '5'])
    with pytest.raises(SystemExit) as error:
        scheduler.main()
    assert error.value.code == 2 and 'pool' in capsys.readouterr().err
