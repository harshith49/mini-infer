"""Real scheduler traces, cache ownership, admission fairness, and retry."""
import copy

import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model
from engine.generate import generate
from engine.sampler import SamplingParams


@pytest.fixture
def model():
    torch.manual_seed(31)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()


@pytest.fixture
def scheduler(model):
    from engine.scheduler import Scheduler
    return Scheduler(model, max_batch_size=2, pad_token_id=0)


def drain(scheduler):
    events = []
    for _ in range(100):
        if scheduler.idle:
            return events
        events.extend(scheduler.step())
    pytest.fail('finite workload did not drain')


def test_fifo_replacement_and_independent_outputs(model, scheduler):
    jobs = [('A', [1], 4), ('B', [2, 3, 4], 1), ('C', [5], 2)]
    for name, prompt, budget in jobs:
        scheduler.submit(name, torch.tensor(prompt), budget)
    first = scheduler.step()
    assert [e.request_id for e in first] == ['A', 'B']
    assert first[0].finish_reason is None and first[1].finish_reason == 'length'
    assert scheduler.result('B').shape == (4,)
    assert [e.request_id for e in scheduler.step()] == ['C', 'A']
    drain(scheduler)
    for name, prompt, budget in jobs:
        assert torch.equal(scheduler.result(name), generate(model, torch.tensor([prompt]), budget, use_cache=True)[0])
    assert scheduler.idle and scheduler.step() == [] and scheduler.cache_allocated_bytes == 0


@pytest.mark.parametrize('limit', [1, 2])
def test_finite_queue_no_starvation_and_dynamic_submission(model, limit):
    from engine.scheduler import Scheduler
    s = Scheduler(model, max_batch_size=limit, pad_token_id=0)
    s.submit('0', torch.tensor([1]), 3)
    events = s.step()
    for i in range(1, 8):
        s.submit(str(i), torch.tensor([i+1]), 1+i%3)
    while not s.idle:
        events.extend(s.step())
        assert len(s._running) <= limit
        assert len(events) <= 24
    seen = list(dict.fromkeys(e.request_id for e in events))
    assert seen == [str(i) for i in range(8)]
    assert sum(e.finish_reason is not None for e in events) == 8


def test_zero_requests_do_no_work_or_consume_slots(model, scheduler, monkeypatch):
    from engine import scheduler as module
    def forbidden(*args, **kwargs):
        raise AssertionError('zero request did allocation or forward')
    monkeypatch.setattr(module, 'SimpleKVCache', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    for name in ['A', 'B', 'C']:
        scheduler.submit(name, torch.tensor([1]), 0)
    events = scheduler.step()
    assert [(e.request_id, e.token_id, e.finish_reason) for e in events] == [(name, None, 'length') for name in ['A', 'B', 'C']]
    assert scheduler.idle and scheduler.cache_allocated_bytes == 0 and scheduler.peak_kv_bytes == 0


def test_zero_waiter_does_not_displace_normal_admission(scheduler):
    scheduler.submit('zero', torch.tensor([1]), 0)
    scheduler.submit('A', torch.tensor([2]), 2)
    scheduler.submit('B', torch.tensor([3]), 2)
    events = scheduler.step()
    assert [e.request_id for e in events] == ['zero', 'A', 'B']
    assert len(scheduler._running) == 2


@pytest.mark.parametrize('budget', [1, 5])
def test_multiple_stop_ids_prompt_stops_and_reason_precedence(model, scheduler, budget):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    scheduler.submit('A', torch.tensor([0]), budget, stop_token_ids=(1, 0))
    assert not scheduler.idle
    event, = scheduler.step()
    assert event.token_id == 0 and event.finish_reason == 'stop'
    assert scheduler.result('A').tolist() == [0, 0]
    assert scheduler._requests['A'].cache is None and scheduler.cache_allocated_bytes == 0


@pytest.mark.parametrize('kwargs', [
    {'request_id': ''}, {'request_id': 1}, {'prompt': torch.tensor([])},
    {'prompt': torch.tensor([[1]])}, {'prompt': torch.tensor([1.])},
    {'prompt': torch.tensor([37])}, {'prompt': torch.tensor([-1])},
    {'prompt': torch.ones(1, dtype=torch.long, device='meta')},
    {'max_new_tokens': -1}, {'max_new_tokens': 16}, {'max_new_tokens': 1.5}, {'max_new_tokens': True},
    {'stop_token_ids': (37,)}, {'stop_token_ids': (-1,)}, {'stop_token_ids': (1.5,)}, {'stop_token_ids': (True,)},
    {'sampling': SamplingParams(temperature=-1)}, {'sampling': SamplingParams(top_k=38)}, {'sampling': 'bad'},
])
def test_bad_submission_does_not_change_state(scheduler, kwargs):
    with pytest.raises(ValueError):
        scheduler.submit(**({'request_id': 'A', 'prompt': torch.tensor([1]), 'max_new_tokens': 1} | kwargs))
    assert scheduler.idle and not scheduler._requests and scheduler.cache_allocated_bytes == 0


@pytest.mark.parametrize('batch,pad', [(0, 0), (-1, 0), (True, 0), (1.5, 0), (2, -1), (2, 37), (2, 0.5)])
def test_invalid_constructor(model, batch, pad):
    from engine.scheduler import Scheduler
    with pytest.raises(ValueError):
        Scheduler(model, max_batch_size=batch, pad_token_id=pad)


def test_duplicate_ids_result_lookup_and_defensive_copies(scheduler):
    prompt = torch.tensor([1, 2])
    scheduler.submit('A', prompt, 0)
    prompt.fill_(7)
    with pytest.raises(ValueError):
        scheduler.submit('A', torch.tensor([3]), 1)
    with pytest.raises(KeyError):
        scheduler.result('unknown')
    with pytest.raises(ValueError):
        scheduler.result('A')
    scheduler.step()
    assert scheduler.result('A').tolist() == [1, 2]
    scheduler.result('A').fill_(8)
    assert scheduler.result('A').tolist() == [1, 2]
    with pytest.raises(ValueError):
        scheduler.submit('A', torch.tensor([3]), 1)


@torch.inference_mode()
def test_packed_logits_real_prefixes_and_poisoned_padding(model, scheduler, monkeypatch):
    from engine import scheduler as module
    from engine.kv_cache import SimpleKVCache
    oracle = copy.deepcopy(model)
    class PoisonedCache(SimpleKVCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.keys.fill_(float('nan'))
            self.values.fill_(float('nan'))
    monkeypatch.setattr(module, 'SimpleKVCache', PoisonedCache)
    phase = []
    for method in ['_prefill', '_decode']:
        original = getattr(scheduler, method)
        def wrapped(requests, original=original, method=method):
            phase[:] = [method, requests]
            return original(requests)
        monkeypatch.setattr(scheduler, method, wrapped)
    observations = []
    def check(module, args, logits):
        assert torch.isfinite(logits).all()
        method, requests = phase
        observations.append((method, len(requests)))
        for row, request in enumerate(requests):
            if method == '_prefill':
                expected = oracle(request.prompt[None])[0]
                actual = logits[row, -len(request.prompt):]
            else:
                expected = oracle(request.output[None])[0, -1:]
                actual = logits[row]
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    hook = model.register_forward_hook(check)
    for name, prompt, budget in [('A', [1], 4), ('B', [2, 3, 4], 3), ('C', [5, 6], 3)]:
        scheduler.submit(name, torch.tensor(prompt), budget)
    try:
        while not scheduler.idle:
            scheduler.step()
            for request in scheduler._requests.values():
                if request.cache is not None:
                    length = len(request.output)-1
                    assert request.cache.length == length and not request.cache.requires_attention_mask
                    expected = SimpleKVCache(model.config, batch_size=1, capacity=length,
                        device=torch.device('cpu'), dtype=torch.float32)
                    oracle(request.output[:-1][None], cache=expected)
                    torch.testing.assert_close(request.cache.keys[:, :, :, :length], expected.keys, atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(request.cache.values[:, :, :, :length], expected.values, atol=1e-4, rtol=1e-4)
    finally:
        hook.remove()
    assert ('_decode', 2) in observations and ('_prefill', 1) in observations


def test_literal_private_and_peak_kv_bytes(scheduler):
    scheduler.submit('A', torch.tensor([1]), 4)
    scheduler.submit('B', torch.tensor([2, 3, 4]), 3)
    scheduler.step()
    assert scheduler.cache_allocated_bytes == 4224
    assert scheduler.peak_kv_bytes == 6528
    drain(scheduler)
    assert scheduler.peak_kv_bytes == 8064
    assert scheduler.cache_allocated_bytes == 0


def saved_state(s):
    return {name: (r.output.clone(), r.generator.get_state().clone(),
        None if r.cache is None else (r.cache.length, r.cache.keys[:, :, :, :r.cache.length].clone(), r.cache.values[:, :, :, :r.cache.length].clone()))
        for name, r in s._requests.items()}


def assert_state(s, before):
    for name, (output, rng, cache) in before.items():
        r = s._requests[name]
        assert torch.equal(r.output, output) and torch.equal(r.generator.get_state(), rng)
        if cache is not None:
            length, keys, values = cache
            assert r.cache.length == length
            torch.testing.assert_close(r.cache.keys[:, :, :, :length], keys, atol=0, rtol=0)
            torch.testing.assert_close(r.cache.values[:, :, :, :length], values, atol=0, rtol=0)


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_failed_forward_preserves_state_and_retry(model, scheduler, monkeypatch, phase):
    scheduler.submit('A', torch.tensor([1]), 4, sampling=SamplingParams(temperature=1., seed=7))
    scheduler.submit('B', torch.tensor([2, 3]), 3)
    if phase == 'decode':
        scheduler.step()
    before = saved_state(scheduler)
    waiting = list(scheduler._waiting)
    original = model.lm_head.forward
    def fail(*args, **kwargs):
        raise RuntimeError('injected projection fault')
    monkeypatch.setattr(model.lm_head, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        scheduler.step()
    assert_state(scheduler, before)
    assert list(scheduler._waiting) == waiting
    monkeypatch.setattr(model.lm_head, 'forward', original)
    drain(scheduler)
    assert len(scheduler.result('A')) == 5 and len(scheduler.result('B')) == 5
    assert scheduler.cache_allocated_bytes == 0


def test_admission_events_survive_later_decode_fault(model, scheduler, monkeypatch):
    for name, prompt, budget in [('A', [1], 4), ('B', [2, 3], 1), ('C', [4], 2)]:
        scheduler.submit(name, torch.tensor(prompt), budget)
    events = scheduler.step()
    a_before = saved_state(scheduler)['A']
    original = model.lm_head.forward
    calls = 0
    def fail_second(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected old decode fault')
        return original(x)
    monkeypatch.setattr(model.lm_head, 'forward', fail_second)
    with pytest.raises(RuntimeError, match='injected'):
        scheduler.step()
    assert_state(scheduler, {'A': a_before})
    assert len(scheduler._requests['C'].output) == 2
    monkeypatch.setattr(model.lm_head, 'forward', original)
    retry = scheduler.step()
    assert [e.request_id for e in retry] == ['C', 'A', 'C']
    events += retry + drain(scheduler)
    assert [e.request_id for e in events].count('C') == 2
    assert [e.request_id for e in events].count('A') == 4
    for name, prompt, budget in [('A', [1], 4), ('B', [2, 3], 1), ('C', [4], 2)]:
        assert torch.equal(scheduler.result(name), generate(model, torch.tensor([prompt]), budget, use_cache=True)[0])


def test_exact_context_budgets_and_one_token_no_persistent_cache(model, scheduler):
    scheduler.submit('A', torch.tensor([1]), 15)
    scheduler.submit('B', torch.tensor([2, 3]), 14)
    drain(scheduler)
    for name, prompt, count in [('A', [1], 15), ('B', [2, 3], 14)]:
        assert len(scheduler.result(name)) == 16
        assert torch.equal(scheduler.result(name), generate(model, torch.tensor([prompt]), count, use_cache=True)[0])
    scheduler.submit('C', torch.tensor([4, 5]), 1)
    event, = scheduler.step()
    assert event.finish_reason == 'length' and scheduler._requests['C'].cache is None
    assert scheduler.cache_allocated_bytes == 0
