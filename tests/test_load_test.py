"""Protocol failures stay visible and client metrics use complete workloads."""
import asyncio
import json

import httpx
import pytest


def sse(events):
    return ''.join('event: '+kind+'\ndata: '+json.dumps(data,ensure_ascii=False)+'\n\n' for kind,data in events).encode()


def events(budget=2, request_id='one'):
    result=[('token',dict(request_id=request_id,token_id=i+1,delta='é')) for i in range(budget)]
    result.append(('done',dict(request_id=request_id,finish_reason='length',token_ids=list(range(1,budget+1)),
        text='A'+'é'*budget,usage=dict(prompt_tokens=1,completion_tokens=budget,total_tokens=1+budget),
        server_peak_forward_batch_size=2)))
    return result


class Chunks(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data=data
    async def __aiter__(self):
        for i in range(0,len(self.data),3):
            yield self.data[i:i+3]


@pytest.mark.parametrize('budget',[0,2])
def test_incremental_sse_validates_ids_deltas_usage_and_terminal(budget):
    from benchmarks.load_test import read_generation
    async def check():
        response=httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(sse(events(budget))))
        result=await read_generation(response,prompt='A',max_new_tokens=budget,started=0)
        assert result['completion_tokens']==budget
        assert (result['ttft_seconds'] is None)==(budget==0)
        assert result['server_peak_forward_batch_size']==2
    asyncio.run(check())


@pytest.mark.parametrize('mutation',['missing','duplicate','wrong_id','usage','ids','text','error','json','trailing','short_length'])
def test_malformed_streams_fail(mutation):
    from benchmarks.load_test import read_generation
    sequence=events()
    if mutation=='missing': sequence.pop()
    if mutation=='duplicate': sequence.append(sequence[-1])
    if mutation=='wrong_id': sequence[1][1]['request_id']='other'
    if mutation=='usage': sequence[-1][1]['usage']['total_tokens']=99
    if mutation=='ids': sequence[-1][1]['token_ids']=[3,4]
    if mutation=='text': sequence[-1][1]['text']='broken'
    if mutation=='error': sequence=[('error',dict(request_id='one',code='worker_failed'))]
    if mutation=='trailing': sequence.append(sequence[0])
    if mutation=='short_length': sequence=events(1)
    data=sse(sequence) if mutation!='json' else b'event: token\ndata: nope\n\n'
    async def check():
        response=httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(data))
        with pytest.raises(ValueError):
            await read_generation(response,prompt='A',max_new_tokens=2,started=0)
    asyncio.run(check())


def test_non200_wrong_content_type_and_truncated_frame():
    from benchmarks.load_test import read_generation
    async def check():
        for response in [httpx.Response(429,json={'detail':'full'}),httpx.Response(200,text='no SSE'),
                         httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(sse(events())[:-2]))]:
            with pytest.raises(ValueError):
                await read_generation(response,prompt='A',max_new_tokens=2,started=0)
    asyncio.run(check())


def test_load_metrics_use_all_requests_and_common_elapsed(monkeypatch):
    from benchmarks import load_test as module
    original=httpx.AsyncClient
    calls=0
    active=peak=0
    async def reply(request):
        nonlocal calls,active,peak
        calls+=1
        request_id=str(calls)
        active+=1
        peak=max(peak,active)
        await asyncio.sleep(0)
        active-=1
        budget=json.loads(request.content)['max_new_tokens']
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(sse(events(budget,request_id))))
    monkeypatch.setattr(module.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(reply),**kw))
    result=asyncio.run(module.run_load('http://test',requests=8,concurrency=4,prompt='A',max_new_tokens=2))
    assert calls==9 and peak<=4
    assert result['requests']==8 and result['completion_tokens']==16 and result['failures']==0
    assert result['tokens_per_second']==pytest.approx(16/result['elapsed_seconds'])
    assert result['elapsed_seconds']>=result['latency_p95_seconds']
    assert result['ttft_samples']==8
    assert module.percentile([1,2,3,4],.5)==2.5
    assert module.percentile([1,2,3,4],.95)==pytest.approx(3.85)


def test_zero_ttft_and_failed_run_do_not_publish_csv(monkeypatch,tmp_path,capsys):
    from benchmarks import load_test as module
    import sys
    original=httpx.AsyncClient
    calls=0
    async def zero(request):
        nonlocal calls
        calls+=1
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(sse(events(0,str(calls)))))
    monkeypatch.setattr(module.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(zero),**kw))
    result=asyncio.run(module.run_load('http://test',requests=2,concurrency=2,prompt='A',max_new_tokens=0))
    assert result['ttft_samples']==0 and result['ttft_p95_seconds'] is None
    async def bad(request):
        return httpx.Response(503,json={'detail':'unavailable'})
    monkeypatch.setattr(module.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(bad),**kw))
    path=tmp_path/'result.csv'
    monkeypatch.setattr(sys,'argv',['load','--url','http://test','--output',str(path),
        '--server-device','cpu','--server-hardware','test','--server-threads','1',
        '--server-max-batch-size','2','--server-cache-backend','contiguous'])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code!=0 and not path.exists()
    assert 'failed_runs=1' in capsys.readouterr().err


@pytest.mark.parametrize('kwargs',[dict(requests=0),dict(concurrency=0),dict(timeout=0),
    dict(timeout=float('inf')),dict(max_new_tokens=-1),dict(requests=True)])
def test_invalid_load_settings_reject_before_http(kwargs):
    from benchmarks.load_test import run_load
    with pytest.raises(ValueError):
        asyncio.run(run_load('http://test',**kwargs))


def test_duplicate_response_identity_and_total_timeout(monkeypatch):
    from benchmarks import load_test as module
    original=httpx.AsyncClient
    async def duplicate(request):
        return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Chunks(sse(events())))
    monkeypatch.setattr(module.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(duplicate),**kw))
    with pytest.raises(ValueError,match='Duplicate'):
        asyncio.run(module.run_load('http://test',requests=2,prompt='A',max_new_tokens=2))
    async def stuck(request):
        await asyncio.Event().wait()
    monkeypatch.setattr(module.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(stuck),**kw))
    with pytest.raises(TimeoutError):
        asyncio.run(module.run_load('http://test',timeout=.01))


def test_csv_append_requires_same_header(monkeypatch,tmp_path):
    from benchmarks import load_test as module
    import csv
    import sys
    async def fake(*args,**kwargs):
        return dict(requests=8,completion_tokens=256,tokens_per_second=80.,failures=0)
    monkeypatch.setattr(module,'run_load',fake)
    path=tmp_path/'result.csv'
    monkeypatch.setattr(sys,'argv',['load','--output',str(path),'--server-device','cpu',
        '--server-hardware','test','--server-threads','1','--server-max-batch-size','2',
        '--server-cache-backend','contiguous'])
    module.main()
    module.main()
    with path.open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[0]['server_hardware']=='test'
    path.write_text('wrong,header\n')
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code==1 and path.read_text()=='wrong,header\n'
