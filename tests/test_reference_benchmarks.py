"""HF generation options and honest optional-backend failure boundaries."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from benchmarks import bench_vs_hf as comparison
from benchmarks import bench_vs_vllm as optional
from benchmarks.stage_suite import Workload
from engine.config import ModelConfig
from engine.model import GPT2Model
from engine.weights import copy_hf_weights


def test_hf_exact_fixed_budgets_with_eos_config():
    torch.manual_seed(3)
    ref = GPT2LMHeadModel(GPT2Config(vocab_size=37, n_positions=16, n_embd=24,
        n_layer=2, n_head=4, n_inner=96, attn_implementation='eager')).eval()
    own = GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()
    copy_hf_weights(own, ref.state_dict())
    prompts = [torch.tensor([1,2]), torch.tensor([3,4,5])]
    w = Workload((2,3), (3,2), page_size=2, num_pages=8)
    from engine.generate import generate
    expected = [generate(own, p[None], n, use_cache=True)[0] for p,n in zip(prompts,w.output_budgets)]
    ref.generation_config.eos_token_id = list(range(37))  # Any greedy token would stop without the override.
    result = comparison.run_hf_once(ref, prompts, w)
    assert all(torch.equal(a,b) for a,b in zip(result.outputs,expected))
    assert result.completions[0] <= result.completions[1] <= result.elapsed
    assert result.peak_memory_bytes is None


def test_optional_missing_dependency_has_no_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(optional.importlib.util, 'find_spec', lambda name: None)
    dest = tmp_path/'result.csv'
    optional.main(['--output',str(dest)])
    assert not dest.exists()
    assert 'skip' in capsys.readouterr().out.lower()


def test_optional_no_cuda_has_no_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(optional.importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    dest = tmp_path/'result.csv'
    optional.main(['--output',str(dest)])
    assert not dest.exists()
    assert 'skip' in capsys.readouterr().out.lower()


def test_optional_runtime_fault_is_not_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(optional.importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    def broken(*args, **kwargs):
        raise RuntimeError('installed API failed')
    monkeypatch.setattr(optional, 'run_vllm', broken)
    with pytest.raises(RuntimeError,match='installed API failed'):
        optional.main(['--output',str(tmp_path/'result.csv')])


def test_optional_validates_before_skip(monkeypatch):
    monkeypatch.setattr(optional.importlib.util, 'find_spec', lambda name: None)
    with pytest.raises(SystemExit) as error:
        optional.main(['--output-budgets','-1'])
    assert error.value.code != 0


def test_paired_models_not_resident_together(monkeypatch):
    import weakref
    holders = []
    cfg = GPT2Config(vocab_size=37,n_positions=16,n_embd=24,n_layer=2,n_head=4,n_inner=96,attn_implementation='eager')
    def own_loader(*args):
        torch.manual_seed(9)
        ref = GPT2LMHeadModel(cfg).eval()
        own = GPT2Model(ModelConfig(vocab_size=37,max_positions=16,hidden_size=24,num_layers=2,num_heads=4,intermediate_size=96)).eval()
        copy_hf_weights(own,ref.state_dict())
        holders.append(weakref.ref(own))
        return own,None
    def hf_loader(*args,**kwargs):
        assert holders[0]() is None
        torch.manual_seed(9)
        return GPT2LMHeadModel(cfg).eval()
    monkeypatch.setattr(comparison,'load_model',own_loader)
    monkeypatch.setattr(comparison.GPT2LMHeadModel,'from_pretrained',hf_loader)
    rows = comparison.run_comparison(Workload((2,3),(3,2),page_size=2,num_pages=8),device_name='cpu')
    assert [r['stage'] for r in rows] == ['mini_infer_cached','hf_cached']
    assert len({tuple(r) for r in rows}) == 1

@pytest.mark.parametrize('module',['benchmarks.bench_vs_hf','benchmarks.bench_vs_vllm'])
def test_reference_imports_keep_hf_cache_local(module):
    import os
    import subprocess
    import sys
    env={k:v for k,v in os.environ.items() if k not in ('HF_HOME','TRANSFORMERS_CACHE')}
    result=subprocess.run([sys.executable,'-c',f'import {module}; from huggingface_hub import constants; print(constants.HF_HUB_CACHE)'],env=env,capture_output=True,text=True,check=False)
    assert result.returncode == 0, result.stderr
    assert '.cache/huggingface' not in result.stderr
    assert result.stdout.strip().endswith('/model_cache/hub')
