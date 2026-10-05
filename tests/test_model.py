"""Structural checks catch causal-mask, validation, and weight-tying bugs."""
import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model


@pytest.fixture
def model():
    torch.manual_seed(17)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
                               num_layers=2, num_heads=4, intermediate_size=96)).eval()


def test_shape_dtype_and_tied_embeddings(model):
    logits = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert logits.shape == (2, 3, 37)
    assert logits.dtype == torch.float32
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_future_tokens_do_not_change_prefix_logits(model):
    with torch.inference_mode():
        a = model(torch.tensor([[1, 2, 3, 4]]))
        b = model(torch.tensor([[1, 2, 9, 8]]))
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=0, rtol=0)


@pytest.mark.parametrize('ids', [torch.tensor([1, 2]), torch.tensor([[1.0]]),
    torch.empty((1, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.long),
    torch.tensor([[-1]]), torch.tensor([[37]]), torch.ones((1, 17), dtype=torch.long)])
def test_rejects_invalid_input(model, ids):
    with pytest.raises(ValueError):
        model(ids)


def test_exact_context_limit(model):
    assert model(torch.ones((1, 16), dtype=torch.long)).shape == (1, 16, 37)


@pytest.mark.parametrize('changes', [{'hidden_size': 25}, {'num_heads': 0},
    {'num_layers': 0}, {'max_positions': -1}, {'intermediate_size': 0},
    {'layer_norm_epsilon': 0}, {'activation_function': 'relu'}])
def test_invalid_model_configuration(changes):
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=37, **changes)
