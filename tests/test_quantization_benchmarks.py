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
