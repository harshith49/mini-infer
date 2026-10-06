"""Sampling changes must preserve independent streams and candidate rules."""
import math

import pytest
import torch

from engine import sampler


def draw(logits, *, count=32, seed=7, **kwargs):
    params = sampler.SamplingParams(seed=seed, **kwargs)
    generator = torch.Generator().manual_seed(seed)
    return [sampler.sample(logits, params, generator=generator).item() for _ in range(count)]


def test_greedy_sampling_preserves_generator_state():
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state().clone()
    token = sampler.sample(torch.tensor([4., 4., 1.]), sampler.SamplingParams(seed=7), generator=generator)
    assert token.item() == 0
    assert torch.equal(before, generator.get_state())


def test_seeded_sampling_matches_probability_oracle():
    logits = torch.tensor([2., 1., -1.], dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)
    expected = [torch.multinomial(torch.softmax(logits / 0.7, -1), 1, generator=generator).item() for _ in range(32)]
    assert draw(logits, temperature=0.7) == expected
    assert draw(logits, temperature=0.7) == draw(logits, temperature=0.7)


@pytest.mark.parametrize('options,allowed', [
    ({'top_p': 0.7}, {0, 1}), ({'top_k': 1}, {0}),
    ({'top_k': 0, 'top_p': 1.0}, {0, 1, 2}),
    ({'top_k': 2, 'top_p': 0.7}, {0, 1}),
    ({'top_k': 2, 'top_p': 0.5}, {0}),
])
def test_filter_candidates(options, allowed):
    logits = torch.tensor([math.log(0.6), math.log(0.3), math.log(0.1)], dtype=torch.float64)
    actual = draw(logits, temperature=1.0, count=128, **options)
    assert set(actual) == allowed


@pytest.mark.parametrize('logits,temperature', [
    (torch.tensor([1., 0., -1.]), 1e-320),
    (torch.tensor([1e308, -1e308], dtype=torch.float64), 1e-320),
    (torch.tensor([1., 0., -1.]), 1e308),
])
def test_extreme_finite_scores_keep_valid_distribution(logits, temperature):
    result = draw(logits, temperature=temperature)
    assert all(0 <= token < len(logits) for token in result)
    if temperature < 1:
        assert set(result) == {0}


@pytest.mark.parametrize('kwargs', [
    {'temperature': -1}, {'temperature': float('nan')}, {'temperature': float('inf')},
    {'temperature': 'x'}, {'top_k': -1}, {'top_k': 4}, {'top_k': 1.5}, {'top_k': True},
    {'top_p': 0}, {'top_p': 1.1}, {'top_p': float('nan')}, {'top_p': float('inf')},
    {'seed': -1}, {'seed': 2**63}, {'seed': 1.5}, {'seed': True},
])
def test_invalid_settings_rejected_before_rng_draw(kwargs):
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        sampler.sample(torch.tensor([1., 2., 3.]), sampler.SamplingParams(**kwargs), generator=generator)
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize('logits', [torch.tensor([]), torch.ones(1, 3), torch.tensor([1, 2]),
    torch.tensor([1., float('nan')]), torch.tensor([1., float('inf')]), torch.tensor([float('-inf'), 1.])])
def test_invalid_logits_rejected_before_rng_draw(logits):
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        sampler.sample(logits, sampler.SamplingParams(temperature=1.), generator=generator)
    assert torch.equal(before, generator.get_state())
