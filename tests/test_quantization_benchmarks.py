"""Sample-integrity and independent sliding-window quality oracles."""
import hashlib
import io
import math
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.nn import functional as F

from engine.config import ModelConfig
from engine.model import GPT2Model


def sample_setup(monkeypatch):
    from benchmarks import bench_quantization as bench
    payload=b'public sample\n'
    monkeypatch.setattr(bench,'SAMPLE_SHA256',hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(bench,'SAMPLE_BYTES',len(payload))
    monkeypatch.delenv('HF_HUB_OFFLINE',raising=False)
    return bench,payload


def test_sample_verified_cache_reuse_and_offline(monkeypatch,tmp_path):
    bench,payload=sample_setup(monkeypatch)
    calls=[]
    def fetch(url,timeout):
        calls.append((url,timeout));return io.BytesIO(payload)
    monkeypatch.setattr(bench,'urlopen',fetch)
    assert bench.load_quality_text(tmp_path)==payload.decode()
    assert len(calls)==1 and 0<calls[0][1]<=30
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    assert bench.load_quality_text(tmp_path)==payload.decode() and len(calls)==1
    with pytest.raises(OSError,match='download|offline|cache'):
        bench.load_quality_text(tmp_path/'missing')


@pytest.mark.parametrize('kind',['checksum','oversized','interrupted','utf8'])
def test_bad_download_never_publishes(monkeypatch,tmp_path,kind):
    bench,payload=sample_setup(monkeypatch)
    data=b'x'*len(payload) if kind=='checksum' else payload+b'extra'
    if kind=='utf8':
        data=b'\xff'*len(payload)
        monkeypatch.setattr(bench,'SAMPLE_SHA256',hashlib.sha256(data).hexdigest())
    class Broken(io.BytesIO):
        def read(self,*args): raise OSError('interrupted')
    monkeypatch.setattr(bench,'urlopen',lambda *a,**kw: Broken(payload) if kind=='interrupted' else io.BytesIO(data))
    with pytest.raises((ValueError,OSError,UnicodeError)): bench.load_quality_text(tmp_path)
    assert not list(tmp_path.rglob('*.txt')) and not list(tmp_path.rglob('*.tmp'))


def test_corrupt_cache_fails_without_network(monkeypatch,tmp_path):
    bench,payload=sample_setup(monkeypatch)
    monkeypatch.setattr(bench,'urlopen',lambda *a,**kw:io.BytesIO(payload))
    bench.load_quality_text(tmp_path)
    path=next(tmp_path.rglob('*.txt'));path.write_bytes(b'corrupt')
    monkeypatch.setattr(bench,'urlopen',lambda *a,**kw:pytest.fail('corrupt cache must fail clearly'))
    with pytest.raises(ValueError,match='checksum|sample|size'):bench.load_quality_text(tmp_path)


def test_quality_prefix_has_no_special_tokens():
    from benchmarks.bench_quantization import quality_input_ids
    calls=[]
    def tokenizer(text,**kwargs):
        calls.append(kwargs); return {'input_ids':list(range(20))}
    ids=quality_input_ids(tokenizer,'sample',target_tokens=13)
    assert torch.equal(ids,torch.arange(14)) and ids.dtype==torch.long and ids.device.type=='cpu'
    assert calls[0]['add_special_tokens'] is False
    with pytest.raises(ValueError):quality_input_ids(tokenizer,'sample',target_tokens=20)
    for invalid in [0,-1,True,1.5]:
        with pytest.raises(ValueError):quality_input_ids(tokenizer,'sample',target_tokens=invalid)


def quality_model():
    return GPT2Model(ModelConfig(vocab_size=37,max_positions=16,hidden_size=24,
        num_layers=2,num_heads=4,intermediate_size=96)).eval()


def test_window_quality_matches_independent_target_oracle(monkeypatch):
    from benchmarks.bench_quantization import evaluate_perplexity
    model=quality_model(); ids=torch.arange(14); calls=[];refs=[]
    def logits_for(tokens):
        context=tokens.cumsum(-1).float()
        return torch.arange(37).float()[None,None,:]*(context[:,:,None]/200)
    def forward(tokens):
        assert all(ref() is None for ref in refs)
        calls.append(tokens.clone()); out=logits_for(tokens);refs.append(weakref.ref(out));return out
    monkeypatch.setattr(model,'forward',forward)
    expected_loss=0.
    # Literal global target/window tuples independently pin final partial block and overlap.
    for a,b,start in [(1,4,0),(4,7,0),(7,10,2),(10,13,5),(13,14,6)]:
        tokens=ids[start:b][None]; logits=logits_for(tokens)
        for target in range(a,b):
            expected_loss+=float(F.cross_entropy(logits[0,target-start-1].double()[None],ids[target:target+1],reduction='sum'))
    result=evaluate_perplexity(model,ids,context=8,stride=3)
    assert result['scored_tokens']==13
    assert result['nll']==pytest.approx(expected_loss/13,abs=1e-7)
    assert result['perplexity']==pytest.approx(math.exp(expected_loss/13),rel=1e-7)
    assert [row[0].tolist() for row in calls]==[list(range(4)),list(range(7)),list(range(2,10)),list(range(5,13)),list(range(6,14))]
    assert all(ref() is None for ref in refs)


def test_uniform_logits_quality_is_vocabulary_size(monkeypatch):
    from benchmarks.bench_quantization import evaluate_perplexity
    model=quality_model()
    monkeypatch.setattr(model,'forward',lambda ids:torch.zeros(1,ids.shape[1],37))
    result=evaluate_perplexity(model,torch.arange(14),context=8,stride=3)
    assert result['perplexity']==pytest.approx(37,rel=1e-7) and result['scored_tokens']==13


@pytest.mark.parametrize('ids,context,stride',[(torch.tensor([1]),8,3),(torch.tensor([1.,2]),8,3),(torch.tensor([[1,2]]),8,3),(torch.tensor([1,37]),8,3),(torch.tensor([1,2]),1,1),(torch.tensor([1,2]),8,8),(torch.tensor([1,2]),17,3),(torch.tensor([1,2]),8,0),(torch.tensor([1,2]),True,1)])
def test_invalid_quality_rejects_before_forward(monkeypatch,ids,context,stride):
    from benchmarks.bench_quantization import evaluate_perplexity
    model=quality_model();monkeypatch.setattr(model,'forward',lambda ids:pytest.fail('invalid input forwarded'))
    with pytest.raises(ValueError):evaluate_perplexity(model,ids,context=context,stride=stride)


def test_nonfinite_quality_fails(monkeypatch):
    from benchmarks.bench_quantization import evaluate_perplexity
    model=quality_model();monkeypatch.setattr(model,'forward',lambda ids:torch.full((1,len(ids[0]),37),float('nan')))
    with pytest.raises(ValueError,match='finite'):evaluate_perplexity(model,torch.tensor([1,2,3]),context=8,stride=3)


def test_comparison_pairs_workloads_storage_and_quality(monkeypatch):
    from benchmarks import bench_quantization as bench
    from engine.quantize import Int8Linear
    from benchmarks.bench_stages import run_workload
    monkeypatch.setattr(bench,'SAMPLE_SHA256',hashlib.sha256(b'sample').hexdigest())
    monkeypatch.setattr(bench,'SAMPLE_BYTES',6)
    model=quality_model(); original=id(model); seen=[]
    def tokenizer(text,**kw):return {'input_ids':[1,2,3,4,5,6,7,8,9,10,11,12,13,14]}
    def observed(current,ids,**kwargs):
        assert id(current)==original
        seen.append((isinstance(current.blocks[0].attention.qkv,Int8Linear),ids.clone()))
        row=run_workload(current,ids,**kwargs)
        row['tokens_per_second']=10 if not seen[-1][0] else 5
        return row
    monkeypatch.setattr(bench,'run_workload',observed,raising=False)
    rows=bench.run_comparison(model,tokenizer,text='sample',prompt_lengths=[2,4],max_new_tokens=3,
        repetitions=3,target_tokens=13,context=8,stride=3)
    assert len(rows)==4 and [converted for converted,_ in seen]==[False,False,True,True]
    assert all(torch.equal(seen[i][1],seen[i+2][1]) for i in [0,1])
    by_stage={stage:[row for row in rows if row['stage']==stage] for stage in ['fp32_cached','int8_cached']}
    for fp,q in zip(by_stage['fp32_cached'],by_stage['int8_cached']):
        assert fp['throughput_ratio']==1 and q['throughput_ratio']==.5
        assert fp['weight_bytes']>q['weight_bytes']
        assert q['model_bytes']>q['weight_bytes'] and q['max_reconstruction_bytes']==9216
        assert fp['cache_allocated_bytes']==q['cache_allocated_bytes']
        assert fp['peak_memory_bytes'] is None and fp['peak_memory_kind']=='unmeasured'
        assert fp['scored_tokens']==q['scored_tokens']==13
        assert fp['token_sha256']==q['token_sha256']
        assert q['relative_ppl_delta']==pytest.approx(q['perplexity']/fp['perplexity']-1)
        assert q['weight_reduction_fraction']==pytest.approx(1-q['weight_bytes']/fp['weight_bytes'])
        assert 'tied' in q['quantization_scope']


@pytest.mark.parametrize('kwargs',[{'prompt_lengths':[]},{'prompt_lengths':[0]},{'prompt_lengths':[15]},
    {'prompt_lengths':[2],'repetitions':2},{'prompt_lengths':[2],'max_new_tokens':0},
    {'prompt_lengths':[2],'stride':8},{'prompt_lengths':[True]}, {'prompt_lengths':[2],'target_tokens':0}])
def test_bad_comparison_rejects_before_conversion(monkeypatch,kwargs):
    from benchmarks import bench_quantization as bench
    model=quality_model()
    monkeypatch.setattr(bench,'quantize_model',lambda *a:pytest.fail('invalid comparison converted'),raising=False)
    settings=dict(max_new_tokens=3,repetitions=3,target_tokens=13,context=8,stride=3);settings.update(kwargs)
    with pytest.raises(ValueError):bench.run_comparison(model,lambda *a,**kw:{'input_ids':list(range(14))},text='sample',**settings)
    assert isinstance(model.blocks[0].attention.qkv,torch.nn.Linear)


def test_comparison_rejects_already_quantized_model():
    from benchmarks import bench_quantization as bench
    from engine.quantize import quantize_model
    with pytest.raises(ValueError):bench.run_comparison(quantize_model(quality_model()),None,text='sample',prompt_lengths=[2])


def test_cli_csv_scopes_and_quality_failure_does_not_publish(monkeypatch,tmp_path):
    import sys,csv,json
    from benchmarks import bench_quantization as bench
    def tokenizer(text,**kw):return {'input_ids':list(range(14))}
    monkeypatch.setattr(bench,'load_model',lambda *a:(quality_model(),tokenizer),raising=False)
    monkeypatch.setattr(bench,'load_quality_text',lambda *a:'sample')
    monkeypatch.setattr(bench,'SAMPLE_SHA256',hashlib.sha256(b'sample').hexdigest())
    monkeypatch.setattr(bench,'SAMPLE_BYTES',6)
    path=tmp_path/'result.csv'
    argv=['benchmark','--device','cpu','--prompt-lengths','2','--max-new-tokens','3','--quality-tokens','13','--context','8','--stride','3','--output',str(path)]
    monkeypatch.setattr(sys,'argv',argv);bench.main()
    with path.open() as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==2 and rows[0]['peak_memory_bytes']=='' and rows[0]['source_sha256']==bench.SAMPLE_SHA256
    assert json.loads(rows[0]['model_config'])['vocab_size']==37
    assert b'\r\n' not in path.read_bytes()
    # A quality failure must not overwrite an existing valid report.
    before=path.read_bytes();original=bench.evaluate_perplexity
    def fail(model,*a,**kw):
        result=original(model,*a,**kw)
        if not isinstance(model.blocks[0].attention.qkv,torch.nn.Linear):result['perplexity']*=100
        return result
    monkeypatch.setattr(bench,'evaluate_perplexity',fail)
    with pytest.raises((ValueError,SystemExit)):bench.main()
    assert path.read_bytes()==before
