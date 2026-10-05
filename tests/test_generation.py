"""Boundary checks for uncached decoding and the user-facing CLI."""
import sys

import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model
from engine.generate import generate
from engine.sampler import greedy
from engine.weights import resolve_device


@pytest.fixture
def model():
    torch.manual_seed(21)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
                               num_layers=2, num_heads=4, intermediate_size=96)).eval()


def test_greedy_picks_largest_logit():
    assert greedy(torch.tensor([[1., 7., 2.], [8., -1., 3.]])).tolist() == [1, 0]


@pytest.mark.parametrize('use_cache', [False, True])
def test_zero_tokens_returns_prompt(model, use_cache):
    ids = torch.tensor([[1, 2]])
    assert torch.equal(generate(model, ids, 0, use_cache=use_cache), ids)


@pytest.mark.parametrize('ids,count', [(torch.tensor([[1]]), -1),
    (torch.tensor([[1]]), 16), (torch.tensor([[1]]), 1.5),
    (torch.tensor([[1], [2]]), 1), (torch.tensor([1]), 0),
    (torch.tensor([[37]]), 0), (torch.tensor([[1.0]]), 0),
    (torch.empty((1, 0), dtype=torch.long), 0)])
@pytest.mark.parametrize('use_cache', [False, True])
def test_invalid_generation_request(model, ids, count, use_cache):
    with pytest.raises(ValueError):
        generate(model, ids, count, use_cache=use_cache)


@pytest.mark.parametrize('use_cache', [False, True])
def test_exact_context_budget(model, use_cache):
    assert generate(model, torch.tensor([[1]]), 15, use_cache=use_cache).shape == (1, 16)


@pytest.mark.parametrize('use_cache', [False, True])
def test_eos_is_included_and_stops_generation(model, use_cache):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    # Zero logits deterministically choose ID 0; no fake decoding loop needed.
    assert generate(model, torch.tensor([[2]]), 5, eos_token_id=0, use_cache=use_cache).tolist() == [[2, 0]]
    assert generate(model, torch.tensor([[2]]), 5, use_cache=use_cache).tolist() == [[2, 0, 0, 0, 0, 0]]


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('auto') == torch.device('cpu')
    assert resolve_device('cpu') == torch.device('cpu')
    with pytest.raises(ValueError, match='CUDA'):
        resolve_device('cuda')
    with pytest.raises(ValueError, match='device'):
        resolve_device('mps')


@pytest.mark.parametrize('use_cache', [False, True])
def test_cli_empty_prompt_uses_eos_seed(model, monkeypatch, capsys, use_cache):
    from engine import generate as cli

    class Tokenizer:
        eos_token_id = 2

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return 'seed=' + ','.join(str(i) for i in ids)

    monkeypatch.setattr(cli, 'load_model', lambda config: (model, Tokenizer()))
    monkeypatch.setattr(sys, 'argv', ['mini-infer', '--prompt', '', '--max-new-tokens', '0',
                                     '--device', 'cpu'])
    if use_cache:
        sys.argv.append('--use-cache')
    cli.main()
    assert capsys.readouterr().out == '\n'


@pytest.mark.parametrize('use_cache', [False, True])
def test_cli_zero_tokens_preserves_literal_special_token(monkeypatch, capsys, use_cache):
    from engine import generate as cli
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='model_cache')
    model = GPT2Model(ModelConfig(vocab_size=50257, max_positions=16, hidden_size=24,
                                 num_layers=1, num_heads=4, intermediate_size=96)).eval()
    prompt = 'Hello <|endoftext|> world'
    monkeypatch.setattr(cli, 'load_model', lambda config: (model, tokenizer))
    monkeypatch.setattr(sys, 'argv', ['mini-infer', '--prompt', prompt,
                                     '--max-new-tokens', '0', '--device', 'cpu'])
    if use_cache:
        sys.argv.append('--use-cache')
    cli.main()
    assert capsys.readouterr().out == prompt + '\n'


@pytest.mark.parametrize('use_cache', [False, True])
def test_cli_empty_prompt_first_forward_uses_eos(model, monkeypatch, capsys, use_cache):
    from engine import generate as cli
    seen = []
    hook = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].tolist()))

    class Tokenizer:
        eos_token_id = 2

        def decode(self, ids, *, skip_special_tokens):
            return ','.join(str(i) for i in ids)

    monkeypatch.setattr(cli, 'load_model', lambda config: (model, Tokenizer()))
    monkeypatch.setattr(sys, 'argv', ['mini-infer', '--prompt', '', '--max-new-tokens', '1'])
    if use_cache:
        sys.argv.append('--use-cache')
    try:
        cli.main()
    finally:
        hook.remove()
    assert seen == [[[2]]]
    capsys.readouterr()


@pytest.mark.parametrize('use_cache,expected', [(False, [3, 4, 5, 6]), (True, [3, 1, 1, 1])])
def test_generation_forward_lengths(model, use_cache, expected):
    seen = []
    hook = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    try:
        result = generate(model, torch.tensor([[1, 2, 3]]), 4, use_cache=use_cache)
    finally:
        hook.remove()
    assert result.shape == (1, 7)
    assert seen == expected


@pytest.mark.parametrize('use_cache', [False, True])
@pytest.mark.parametrize('count', [0, 1, 3])
def test_step_timings_cover_only_produced_tokens(model, use_cache, count):
    timings = []
    output = generate(model, torch.tensor([[1, 2]]), count,
                      use_cache=use_cache, step_times=timings)
    assert output.shape == (1, 2 + count)
    assert len(timings) == count
    assert all(duration >= 0 for duration in timings)


def test_zero_output_does_not_allocate_or_forward(model, monkeypatch):
    from engine import generate as module

    def forbidden(*args, **kwargs):
        raise AssertionError('zero output did unnecessary work')

    monkeypatch.setattr(module, 'SimpleKVCache', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    assert generate(model, torch.tensor([[1]]), 0, use_cache=True).tolist() == [[1]]


@pytest.mark.parametrize('use_cache', [False, True])
def test_previous_logits_released_before_next_forward(model, use_cache):
    import weakref

    references = []
    released = []
    before = model.register_forward_pre_hook(
        lambda module, args: released.append(not references or references[-1]() is None))
    after = model.register_forward_hook(
        lambda module, args, output: references.append(weakref.ref(output)))
    try:
        generate(model, torch.tensor([[1, 2, 3]]), 4, use_cache=use_cache)
    finally:
        before.remove()
        after.remove()
    assert released == [True, True, True, True]
