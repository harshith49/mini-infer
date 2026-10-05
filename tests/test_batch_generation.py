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



@pytest.mark.parametrize('use_cache', [False, True])
@pytest.mark.parametrize('order', [[0, 1, 2], [2, 0, 1], [1]])
@pytest.mark.parametrize('pad', [0, 1])
def test_batch_matches_independent_requests(model, use_cache, order, pad):
    from engine.batching import generate_batch
    from engine.generate import generate
    source = [torch.tensor([1]), torch.tensor([2, 3, 4]), torch.tensor([5, 6])]
    prompts = [source[i] for i in order]
    actual = generate_batch(model, prompts, 4, pad_token_id=pad, use_cache=use_cache)
    expected = [generate(model, p[None], 4, use_cache=use_cache)[0] for p in prompts]
    assert len(actual) == len(prompts)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize('use_cache,lengths', [(False, [3, 4, 5, 6]), (True, [3, 1, 1, 1])])
def test_batch_forward_lengths_masks_and_logits_lifetime(model, use_cache, lengths):
    import weakref
    from engine.batching import generate_batch
    seen, references = [], []
    def before(module, args, kwargs):
        assert not references or references[-1]() is None
        seen.append((args[0].shape[1], kwargs['attention_mask'].clone()))
    before_hook = model.register_forward_pre_hook(before, with_kwargs=True)
    after_hook = model.register_forward_hook(lambda module, args, output: references.append(weakref.ref(output)))
    try:
        output = generate_batch(model, [torch.tensor([1]), torch.tensor([2, 3, 4])],
                                4, pad_token_id=0, use_cache=use_cache)
    finally:
        before_hook.remove()
        after_hook.remove()
    assert [n for n, mask in seen] == lengths
    assert [len(row) for row in output] == [5, 7]
    for step, (_, mask) in enumerate(seen):
        assert mask.tolist() == [[False, False] + [True] * (1 + step), [True] * (3 + step)]


@pytest.mark.parametrize('use_cache', [False, True])
@pytest.mark.parametrize('all_finished', [False, True])
def test_independent_eos_and_masked_filler(model, monkeypatch, use_cache, all_finished):
    from engine.batching import generate_batch
    masks = []
    def before(module, args, kwargs):
        masks.append(kwargs['attention_mask'].clone())
    hook = model.register_forward_pre_hook(before, with_kwargs=True)
    original = model.lm_head.forward
    calls = 0
    def controlled_projection(x):
        nonlocal calls
        logits = original(x)  # Keep real transformer and vocabulary projection execution.
        logits[:, -1, :] = -1000
        logits[0, -1, 0] = 1000
        logits[1, -1, 0 if all_finished or calls == 2 else 7] = 1000
        calls += 1
        return logits
    monkeypatch.setattr(model.lm_head, 'forward', controlled_projection)
    try:
        result = generate_batch(model, [torch.tensor([0]), torch.tensor([2, 3, 4])],
                                5, pad_token_id=0, eos_token_id=0, use_cache=use_cache)
    finally:
        hook.remove()
    assert result[0].tolist() == [0, 0]  # Real EOS-valued prompt is not padding or stopping.
    assert result[1].tolist() == ([2, 3, 4, 0] if all_finished else [2, 3, 4, 7, 7, 0])
    assert calls == (1 if all_finished else 3)
    if not all_finished:
        assert masks[1].tolist() == [[False, False, True, True], [True] * 4]
        assert masks[2].tolist() == [[False, False, True, True, False], [True] * 5]


@pytest.mark.parametrize('use_cache', [False, True])
def test_batch_zero_output_does_no_work(model, monkeypatch, use_cache):
    from engine import batching
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected allocation or forward')
    monkeypatch.setattr(batching, 'SimpleKVCache', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    prompts = [torch.tensor([1]), torch.tensor([2, 3])]
    result = batching.generate_batch(model, prompts, 0, pad_token_id=0, use_cache=use_cache)
    assert all(a is b for a, b in zip(result, prompts))


@pytest.mark.parametrize('prompts,count,options', [
    ([], 1, {}),
    ([torch.tensor([1]), torch.empty(0, dtype=torch.long)], 0, {}),
    ([torch.tensor([1]), torch.tensor([[2]])], 1, {}),
    ([torch.tensor([1]), torch.tensor([2.])], 1, {}),
    ([torch.tensor([1]), torch.tensor([37])], 1, {}),
    ([torch.tensor([1]), torch.tensor([-1])], 1, {}),
    ([torch.tensor([1]), torch.ones(1, dtype=torch.long, device='meta')], 1, {}),
    ([torch.tensor([1])], -1, {}),
    ([torch.tensor([1])], 1.5, {}),
    ([torch.tensor([1])], 16, {}),
    ([torch.tensor([1])], 1, {'pad_token_id': -1}),
    ([torch.tensor([1])], 1, {'pad_token_id': 37}),
    ([torch.tensor([1])], 1, {'pad_token_id': 0.5}),
    ([torch.tensor([1])], 1, {'eos_token_id': -1}),
    ([torch.tensor([1])], 1, {'eos_token_id': 37}),
    ([torch.tensor([1])], 1, {'eos_token_id': 0.5}),
])
def test_invalid_batch_rejected_before_work(model, monkeypatch, prompts, count, options):
    from engine import batching
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid request did work')
    monkeypatch.setattr(batching, 'SimpleKVCache', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    with pytest.raises(ValueError):
        batching.generate_batch(model, prompts, count, use_cache=True,
                                **({'pad_token_id': 0} | options))


@pytest.mark.parametrize('use_cache', [False, True])
def test_batch_exact_context_budget(model, use_cache):
    from engine.batching import generate_batch
    result = generate_batch(model, [torch.tensor([1]), torch.tensor([2, 3])],
                            14, pad_token_id=0, use_cache=use_cache)
    assert [len(row) for row in result] == [15, 16]
