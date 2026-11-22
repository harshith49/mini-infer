"""Charts reject incomparable rows and never turn absent memory into zero."""
import csv

import pytest
from benchmarks.make_charts import make_charts, read_rows
from benchmarks.stage_suite import STAGES


def rows():
    return [dict(stage=stage,device='cpu',hardware='test CPU',dtype='int8_weights_fp32_compute' if stage=='int8' else 'fp32',threads='1',
        prompt_lengths='[2,5]',output_budgets='[1,3]',request_count='2',max_batch_size='2',seed='0',warmups='1',repetitions='3',
        python_version='3.11.0',torch_version='2.5.1',
        cache_backend='paged' if stage in ('paged_kv','int8') else 'none' if stage=='naive' else 'contiguous',
        page_size='16' if stage in ('paged_kv','int8') else '',num_pages='32' if stage in ('paged_kv','int8') else '',
        workload='synthetic',latency_scope='common_submission',tokens_per_second=str(10+i),
        completion_p50_seconds='1',completion_p95_seconds='2',peak_memory_bytes='',peak_memory_kind='unmeasured') for i,stage in enumerate(STAGES)]


def save(tmp_path, data):
    path=tmp_path/'data.csv'
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(data[0]))
        writer.writeheader();writer.writerows(data)
    return path


def test_valid_pngs_and_missing_memory(tmp_path):
    source=save(tmp_path,rows())
    parsed=read_rows(source)
    assert all(row['peak_memory_bytes'] is None for row in parsed)
    files=make_charts(source,tmp_path/'charts')
    assert len(files)==3
    assert all(p.read_bytes().startswith(b'\x89PNG') for p in files)


@pytest.mark.parametrize('problem', ['duplicate','missing','nan','negative','latency','hardware','budget','memory','memory_kind','missing_column'])
def test_bad_csv_no_charts(tmp_path,problem):
    data=rows()
    if problem=='duplicate': data[-1]['stage']=data[0]['stage']
    elif problem=='missing': data.pop()
    elif problem=='nan': data[0]['tokens_per_second']='nan'
    elif problem=='negative': data[0]['completion_p50_seconds']='-1'
    elif problem=='latency': data[0]['completion_p50_seconds']='3'
    elif problem=='hardware': data[0]['hardware']='other'
    elif problem=='budget': data[0]['output_budgets']='[2,3]'
    elif problem=='memory': data[0]['peak_memory_bytes']='0'
    elif problem=='memory_kind': data[0]['peak_memory_kind']='invented'
    elif problem=='missing_column':
        for row in data: del row['tokens_per_second']
    source=save(tmp_path,data)
    target=tmp_path/'charts'
    with pytest.raises(ValueError): make_charts(source,target)
    assert not target.exists()


def test_unmeasured_cuda_is_not_labeled_cpu(tmp_path,monkeypatch):
    from matplotlib.axes import Axes
    data=rows()
    for row in data: row['device']='cuda'
    captured=[]
    original=Axes.text
    def text(self,x,y,s,*args,**kwargs):
        captured.append(s)
        return original(self,x,y,s,*args,**kwargs)
    monkeypatch.setattr(Axes,'text',text)
    make_charts(save(tmp_path,data),tmp_path/'charts')
    assert any('Unmeasured on cuda' in s for s in captured)
    assert not any('Unmeasured on CPU' in s for s in captured)


@pytest.mark.parametrize('field,value', [('python_version','other'), ('torch_version','999'),
    ('python_version',''), ('num_pages','64'), ('page_size','0'), ('num_pages',''), ('cache_backend','contiguous')])
def test_incompatible_configuration_no_charts(tmp_path, field, value):
    data = rows()
    data[-1][field] = value
    target = tmp_path/'charts'
    with pytest.raises(ValueError):
        make_charts(save(tmp_path,data),target)
    assert not target.exists()


def test_missing_configuration_column(tmp_path):
    data = rows()
    for row in data: del row['page_size']
    with pytest.raises(ValueError):
        read_rows(save(tmp_path,data))
