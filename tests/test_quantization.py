"""Independent int8 arithmetic, source lifetime, and transformer conversion."""
import weakref

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from engine.config import ModelConfig
from engine.model import GPT2Model


def tiny_model():
    torch.manual_seed(11)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=64, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()


def projections(model):
    return [layer for block in model.blocks for layer in
            (block.attention.qkv, block.attention.projection, block.mlp.up, block.mlp.down)]


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('shape', [(3,), (2,3), (2,2,3)])
def test_per_channel_symmetric_values_and_independent_forward(bias, shape):
    from engine.quantize import Int8Linear
    source = nn.Linear(3, 4, bias=bias).eval()
    weight = torch.tensor([[-127.,0,127],[-254,0,254],[0,0,0],[.5,1.5,127]])
    with torch.no_grad():
        source.weight.copy_(weight)
        if bias:
            source.bias.copy_(torch.tensor([1.,2,3,4]))
    layer = Int8Linear.from_linear(source)
    assert layer.qweight.dtype == torch.int8
    assert layer.scale.dtype == torch.float32 and layer.scale.shape == (4,1)
    assert layer.in_features == 3 and layer.out_features == 4 and not layer.training
    assert torch.equal(layer.qweight, torch.tensor([[-127,0,127],[-127,0,127],[0,0,0],[0,2,127]],dtype=torch.int8))
    assert torch.equal(layer.scale, torch.tensor([[1.],[2.],[1.],[1.]]))
    reconstructed = torch.tensor([[-127.,0,127],[-254,0,254],[0,0,0],[0,2,127]])
    assert ((weight-reconstructed).abs() <= layer.scale/2).all()
    x = torch.arange(torch.tensor(shape).prod()).float().reshape(shape)/10
    torch.testing.assert_close(layer(x), F.linear(x,reconstructed,torch.tensor([1.,2,3,4]) if bias else None))
    assert list(layer.parameters()) == []


@pytest.mark.parametrize('kind', ['double','nonlinear','nan','bias_inf'])
def test_invalid_source_rejects(kind):
    from engine.quantize import Int8Linear
    source = nn.Linear(3,2)
    if kind == 'double': source = source.double()
    if kind == 'nonlinear': source = nn.ReLU()
    if kind == 'nan': source.weight.data[0,0] = float('nan')
    if kind == 'bias_inf': source.bias.data[0] = float('inf')
    with pytest.raises(ValueError): Int8Linear.from_linear(source)


@pytest.mark.parametrize('x', [torch.ones(2,2),torch.ones(2,3,dtype=torch.float64),torch.ones(2,3,dtype=torch.long),torch.tensor(1.),torch.empty(2,3,device='meta')])
def test_invalid_input_rejects(x):
    from engine.quantize import Int8Linear
    layer = Int8Linear.from_linear(nn.Linear(3,2))
    with pytest.raises(ValueError): layer(x)


def test_zero_subnormal_and_large_finite_rows():
    from engine.quantize import Int8Linear
    source = nn.Linear(2,3,bias=False)
    source.weight.data.copy_(torch.tensor([[0.,0.],[1e-44,-1e-44],[1e38,-1e38]]))
    layer = Int8Linear.from_linear(source)
    assert torch.isfinite(layer.scale).all() and (layer.scale > 0).all()
    reconstructed = layer.qweight.float()*layer.scale
    assert torch.isfinite(reconstructed).all() and torch.equal(reconstructed[0],torch.zeros(2))
    error = (reconstructed.double()-source.weight.double()).abs()
    assert (error <= layer.scale.double()/2 + source.weight.double().abs()*1e-7).all()


def test_reconstruction_overflow_rejects():
    from engine.quantize import Int8Linear
    source = nn.Linear(1,1,bias=False)
    source.weight.data.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError,match='finite'): Int8Linear.from_linear(source)


def test_conversion_tied_head_idempotence_and_source_release():
    from engine.quantize import Int8Linear, quantize_model
    model = tiny_model()
    refs = [weakref.ref(layer) for layer in projections(model)]
    weight_refs = [weakref.ref(layer.weight) for layer in projections(model)]
    embedding = model.token_embedding.weight
    assert quantize_model(model) is model
    assert all(isinstance(layer,Int8Linear) for layer in projections(model))
    assert len(projections(model)) == 8
    assert model.lm_head.weight is embedding and model.token_embedding.weight is embedding
    assert all(ref() is None for ref in refs+weight_refs)
    pointers = [layer.qweight.data_ptr() for layer in projections(model)]
    assert quantize_model(model) is model
    assert pointers == [layer.qweight.data_ptr() for layer in projections(model)]
    model.to('cpu')
    assert all(layer.qweight.dtype==torch.int8 and layer.scale.dtype==torch.float32 for layer in projections(model))


@pytest.mark.parametrize('kind',['shape','double','nan','mixed'])
def test_late_invalid_topology_leaves_earlier_layers_intact(kind):
    from engine.quantize import Int8Linear,quantize_model
    model=tiny_model(); first=model.blocks[0].attention.qkv
    if kind=='shape': model.blocks[-1].mlp.down=nn.Linear(7,24)
    if kind=='double': model.blocks[-1].mlp.down.double()
    if kind=='nan': model.blocks[-1].mlp.down.weight.data[0,0]=float('nan')
    if kind=='mixed': model.blocks[-1].mlp.down=Int8Linear.from_linear(model.blocks[-1].mlp.down)
    with pytest.raises(ValueError): quantize_model(model)
    assert model.blocks[0].attention.qkv is first


def test_non_model_conversion_rejects():
    from engine.quantize import quantize_model
    with pytest.raises(ValueError): quantize_model(nn.Linear(3,2))


def test_real_storage_bytes_and_no_reconstruction_retention(monkeypatch):
    from engine.quantize import quantize_model,model_storage_bytes,Int8Linear
    model=tiny_model()
    # Parameters: tied vocab embedding888,position1536,norm240,bias432,linear13824 =16920 FP32.
    before=model_storage_bytes(model)
    assert before=={'weight_bytes':67680,'model_bytes':75872,'max_reconstruction_bytes':0}
    quantize_model(model)
    # Int8 matrices13824; scales432,bias432,norm240,embeddings2424 all FP32.
    after=model_storage_bytes(model)
    assert after=={'weight_bytes':27936,'model_bytes':36128,'max_reconstruction_bytes':9216}
    model.register_buffer('embedding_view',model.token_embedding.weight[:, :2])
    assert model_storage_bytes(model)==after
    refs=[]; original=F.linear
    def inspect(x,w,b=None):
        if w.shape in [(72,24),(24,24),(96,24),(24,96)]:
            assert all(ref() is None for ref in refs)
            refs.append(weakref.ref(w))
        return original(x,w,b)
    monkeypatch.setattr(F,'linear',inspect)
    with torch.inference_mode(): model(torch.tensor([[1,2,3]]))
    assert len(refs)==8 and all(ref() is None for ref in refs)
    for layer in projections(model):
        assert isinstance(layer,Int8Linear)
        assert set(layer.state_dict())=={'qweight','scale','bias'}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable')
def test_int8_device_movement_cuda():
    from engine.quantize import quantize_model
    model=quantize_model(tiny_model()).to('cuda')
    with torch.inference_mode(): assert torch.isfinite(model(torch.tensor([[1,2]],device='cuda'))).all()
    assert all(layer.qweight.dtype==torch.int8 and layer.scale.dtype==torch.float32 for layer in projections(model))
