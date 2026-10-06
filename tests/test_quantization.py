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


@pytest.mark.parametrize('value',[1,'yes',None])
def test_int8_config_rejects_non_boolean(value):
    from engine.config import EngineConfig
    with pytest.raises(ValueError): EngineConfig(int8=value)


@pytest.mark.parametrize('enabled',[False,True])
def test_loader_maps_before_quantizing(monkeypatch,enabled):
    from transformers import GPT2Config,GPT2LMHeadModel
    from engine import weights
    from engine.config import EngineConfig
    from engine.quantize import Int8Linear
    hf=GPT2Config(vocab_size=37,n_positions=64,n_embd=24,n_layer=2,n_head=4,n_inner=96)
    reference=GPT2LMHeadModel(hf).eval()
    expected=reference.transformer.h[0].attn.c_attn.weight.detach().T.clone()
    monkeypatch.setattr(weights.GPT2Config,'from_pretrained',lambda *a,**kw:hf)
    monkeypatch.setattr(weights.GPT2LMHeadModel,'from_pretrained',lambda *a,**kw:(reference,{'missing_keys':[],'mismatched_keys':[],'error_msgs':[]}))
    monkeypatch.setattr(weights.AutoTokenizer,'from_pretrained',lambda *a,**kw:object())
    model,_=weights.load_model(EngineConfig(device='cpu',int8=enabled))
    layer=model.blocks[0].attention.qkv
    assert isinstance(layer,Int8Linear)==enabled
    if enabled:
        error=(layer.qweight.float()*layer.scale-expected).abs()
        assert (error<=layer.scale/2+1e-7).all()
    else: torch.testing.assert_close(layer.weight,expected,atol=0,rtol=0)
    assert model.lm_head.weight is model.token_embedding.weight and not model.training


@pytest.mark.parametrize('module_name',['generate','batching','scheduler'])
@pytest.mark.parametrize('enabled',[False,True])
def test_cli_int8_ordered_outputs(monkeypatch,capsys,module_name,enabled):
    import importlib,json,sys
    from engine.quantize import quantize_model
    from engine.generate import generate
    module=importlib.import_module('engine.'+module_name)
    model=tiny_model(); seen=[]
    class Tokenizer:
        eos_token_id=0
        def __call__(self,text,**kw):return {'input_ids':torch.tensor([[ord(c)%36+1 for c in text]])}
        def decode(self,ids,**kw):return ' '.join(str(i) for i in ids if i!=0)
    tok=Tokenizer()
    def load(config):
        seen.append(config.int8)
        return (quantize_model(model) if config.int8 else model),tok
    monkeypatch.setattr(module,'load_model',load)
    argv=[module_name,'--prompt','Hi','--max-new-tokens','3','--device','cpu']
    if enabled:argv+=['--int8']
    if module_name!='scheduler':argv+=['--use-cache']
    if module_name!='generate':argv+=['--prompt','Café']
    if module_name=='scheduler':argv+=['--cache-backend','paged','--num-pages','2','--page-size','4']
    monkeypatch.setattr(sys,'argv',argv);module.main()
    texts=['Hi'] if module_name=='generate' else ['Hi','Café']
    expected=[]
    for text in texts:
        ids=tok(text)['input_ids'];output=generate(model,ids,3,use_cache=True,eos_token_id=0)
        expected.append(text+tok.decode(output[0,len(ids[0]):].tolist()))
    actual=capsys.readouterr().out.strip()
    assert (actual if module_name=='generate' else json.loads(actual))==(expected[0] if module_name=='generate' else expected)
    assert seen==[enabled]


def test_quantized_cache_chunk_batch_and_page_retry(monkeypatch):
    from engine.quantize import quantize_model
    from engine.generate import generate
    from engine.batching import generate_batch
    from engine.kv_cache import SimpleKVCache,PagePool,PagedKVCache
    from engine.scheduler import Scheduler
    model=quantize_model(tiny_model());ids=torch.tensor([[1,2,3,4,5,6,7]])
    pool=PagePool(model.config,num_pages=5,page_size=4,device=torch.device('cpu'),dtype=torch.float32)
    for cache in [SimpleKVCache(model.config,batch_size=1,capacity=16,device=torch.device('cpu'),dtype=torch.float32),PagedKVCache(pool,capacity=16)]:
        if isinstance(cache,PagedKVCache):pool.keys.fill_(float('nan'));pool.values.fill_(float('nan'))
        with torch.inference_mode():
            model(ids[:,:3],cache=cache)
            suffix=model(ids[:,3:],cache=cache)
            torch.testing.assert_close(suffix,model(ids)[:,3:],atol=1e-4,rtol=1e-4)
        if isinstance(cache,PagedKVCache):cache.close()
    prompts=[torch.tensor([1,2,3]),torch.tensor([4,5,6,7,8])]
    expected=[generate(model,p[None],6,use_cache=True)[0] for p in prompts]
    assert all(torch.equal(a,b) for a,b in zip(expected,generate_batch(model,prompts,6,pad_token_id=0,use_cache=True)))
    assert all(torch.equal(a,generate(model,p[None],6)[0]) for a,p in zip(expected,prompts))
    scheduler=Scheduler(model,max_batch_size=2,pad_token_id=0,page_pool=pool)
    for i,p in enumerate(prompts):scheduler.submit(str(i),p,6)
    scheduler.submit('zero',torch.tensor([1]),0)
    scheduler.step();before=scheduler.result('0').clone() if scheduler.idle else scheduler._requests['0'].output.clone()
    original=model.lm_head.forward
    def fail(x):raise RuntimeError('injected late projection')
    monkeypatch.setattr(model.lm_head,'forward',fail)
    with pytest.raises(RuntimeError,match='injected'):scheduler.step()
    assert torch.equal(scheduler._requests['0'].output,before)
    monkeypatch.setattr(model.lm_head,'forward',original)
    while not scheduler.idle:scheduler.step()
    assert all(torch.equal(scheduler.result(str(i)),x) for i,x in enumerate(expected))
    assert torch.equal(scheduler.result('zero'),torch.tensor([1])) and pool.free_pages==5


def test_int8_seeded_sampling_stops_and_global_rng_under_page_pressure():
    from engine.quantize import quantize_model
    from engine.generate import generate
    from engine.kv_cache import PagePool
    from engine.scheduler import Scheduler
    from engine.sampler import SamplingParams
    model = quantize_model(tiny_model())
    prompts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])]
    stop = generate(model, prompts[0][None], 1, use_cache=True)[0, -1].item()
    settings = [SamplingParams(), SamplingParams(temperature=.8, top_k=7, top_p=.9, seed=13)]
    random_state = torch.random.get_rng_state().clone()
    expected = []
    for i, prompt in enumerate(prompts):
        alone = Scheduler(model, max_batch_size=1, pad_token_id=0)
        alone.submit(str(i), prompt, 6, sampling=settings[i], stop_token_ids=(stop,) if i == 0 else ())
        while not alone.idle:
            alone.step()
        expected.append(alone.result(str(i)))
    pool = PagePool(model.config, num_pages=4, page_size=4, device=torch.device('cpu'), dtype=torch.float32)
    mixed = Scheduler(model, max_batch_size=3, pad_token_id=0, page_pool=pool)
    for i, prompt in enumerate(prompts):
        mixed.submit(str(i), prompt, 6, sampling=settings[i], stop_token_ids=(stop,) if i == 0 else ())
    mixed.submit('zero', torch.tensor([9]), 0)
    events = []
    while not mixed.idle:
        events.extend(mixed.step())
    assert len(expected[0]) == len(prompts[0]) + 1
    assert all(torch.equal(mixed.result(str(i)), output) for i, output in enumerate(expected))
    assert torch.equal(mixed.result('zero'), torch.tensor([9]))
    assert [event.request_id for event in events].count('0') == 1
    assert pool.free_pages == 4 and torch.equal(random_state, torch.random.get_rng_state())
