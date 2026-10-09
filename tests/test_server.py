"""Strict HTTP boundary and complete SSE contract via TestClient."""
import json
import sys

import pytest
from fastapi.testclient import TestClient

from test_server_worker import tiny_model, Tokenizer


def decode_events(content):
    events = []
    for block in content.strip().split('\n\n'):
        kind, data = block.split('\n')
        events.append((kind.removeprefix('event: '), json.loads(data.removeprefix('data: '))))
    return events


@pytest.mark.parametrize('change', [dict(prompt=3), dict(prompt=True), dict(prompt='x'*65537),
    dict(extra=1), dict(max_new_tokens=True), dict(max_new_tokens='2'), dict(max_new_tokens=-1),
    dict(max_new_tokens=1.5), dict(max_new_tokens=100), dict(seed=True), dict(seed=-1),
    dict(seed=2**63), dict(top_k=38), dict(top_k='1'), dict(top_p=0), dict(top_p=True),
    dict(temperature=-1), dict(temperature=True), dict(stop_token_ids=[37]),
    dict(stop_token_ids=[True]), dict(stop_token_ids=None)])
def test_generate_schema_and_admission_errors(change):
    from server.app import create_app
    app = create_app(loader=lambda: (tiny_model(), Tokenizer()))
    with TestClient(app) as client:
        response = client.post('/v1/generate', json=dict(prompt='A', max_new_tokens=2) | change)
        assert response.status_code == 422
        assert response.headers['content-type'].startswith('application/json')
        assert client.post('/v1/generate', json=dict(prompt='A', max_new_tokens=0)).status_code == 200


@pytest.mark.parametrize('field,value', [('temperature','NaN'),('temperature','Infinity'),('top_p','NaN')])
def test_nonfinite_validation_returns_json(field, value):
    from server.app import create_app
    with TestClient(create_app(loader=lambda: (tiny_model(), Tokenizer()))) as client:
        raw = '{"prompt":"A","' + field + '":' + value + '}'
        assert client.post('/v1/generate', content=raw, headers={'content-type':'application/json'}).status_code == 422


def test_sse_token_done_and_zero_budget_contract():
    from server.app import create_app
    app = create_app(loader=lambda: (tiny_model(), Tokenizer()))
    with TestClient(app) as client:
        for text, budget in [('A\n',3), ('',0), ('Café',2)]:
            response = client.post('/v1/generate', json=dict(prompt=text,max_new_tokens=budget,stop_token_ids=[]))
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            assert response.headers['cache-control'] == 'no-cache'
            events = decode_events(response.text)
            tokens = [data for kind,data in events if kind == 'token']
            done = events[-1][1]
            assert events[-1][0] == 'done' and len(tokens) == budget
            assert done['text'] == text + ''.join(t['delta'] for t in tokens)
            assert done['token_ids'] == [t['token_id'] for t in tokens]
            assert all(t['request_id'] == done['request_id'] for t in tokens)
            assert done['usage'] == dict(prompt_tokens=max(1,len(text)),completion_tokens=budget,
                                        total_tokens=max(1,len(text))+budget)
        assert not app.state.worker._handles


def test_special_stop_sampling_and_pool_budget():
    from server.app import create_app
    model = tiny_model()
    with TestClient(create_app(loader=lambda: (model,Tokenizer()),cache_backend='paged',num_pages=1,page_size=4)) as c:
        invalid = c.post('/v1/generate',json=dict(prompt='ABCD',max_new_tokens=1))
        assert invalid.status_code == 422
        greedy = decode_events(c.post('/v1/generate',json=dict(prompt='A',max_new_tokens=1,stop_token_ids=[])).text)[0][1]['token_id']
        done = decode_events(c.post('/v1/generate',json=dict(prompt='A',max_new_tokens=3,stop_token_ids=[greedy])).text)[-1][1]
        assert done['finish_reason'] == 'stop' and done['usage']['completion_tokens'] == 1
        data=dict(prompt='A',max_new_tokens=3,temperature=5,top_k=7,top_p=.8,seed=17,stop_token_ids=[])
        assert decode_events(c.post('/v1/generate',json=data).text)[-1][1]['token_ids'] == decode_events(c.post('/v1/generate',json=data).text)[-1][1]['token_ids']


@pytest.mark.parametrize('flags', [['--port','0'],['--threads','0'],['--max-outstanding','0'],['--num-pages','0']])
def test_cli_rejects_bad_settings_before_loading(monkeypatch, flags):
    from server import app
    monkeypatch.setattr(app, 'load_model', lambda *a: pytest.fail('invalid CLI loaded weights'))
    monkeypatch.setattr(sys,'argv',['server']+flags)
    with pytest.raises(SystemExit) as e:
        app.main()
    assert e.value.code == 2


def test_response_header_send_failure_still_disposes_handle():
    import asyncio
    from server.app import GenerationResponse
    from server.worker import GenerationWorker
    from starlette.requests import ClientDisconnect
    from test_server_worker import payload
    async def check():
        worker=GenerationWorker(lambda:(tiny_model(),Tokenizer()))
        await worker.start()
        try:
            handle=await worker.submit(payload(budget=0))
            response=GenerationResponse(worker,handle)
            async def send(message):
                raise OSError('closed transport')
            async def receive():
                await asyncio.Event().wait()
            with pytest.raises(ClientDisconnect):
                await response(dict(type='http',asgi={'spec_version':'2.4'}),receive,send)
            assert not worker._handles
        finally:
            await worker.stop()
    asyncio.run(check())


def test_cli_normal_interrupt_exits_without_traceback(monkeypatch):
    from server import app
    def interrupt(awaitable):
        awaitable.close()
        raise KeyboardInterrupt
    monkeypatch.setattr(app.asyncio,'run',interrupt)
    monkeypatch.setattr(sys,'argv',['server','--device','cpu'])
    try:
        app.main()
    except KeyboardInterrupt:
        pytest.fail('Normal server interrupt escaped the CLI')


@pytest.mark.parametrize('spec_version', ['2.3', '2.4'])
@pytest.mark.parametrize('completed', [False, True])
def test_shutdown_preserves_terminal_for_paused_response(spec_version, completed):
    import asyncio
    import threading
    from server.app import GenerationResponse
    from server.worker import GenerationWorker
    from test_server_worker import payload
    model = tiny_model()
    entered, unblock = threading.Event(), threading.Event()
    calls = 0
    def block_decode(module, args):
        nonlocal calls
        calls += 1
        if not completed and calls == 2:
            entered.set()
            assert unblock.wait(5)
    hook = model.register_forward_pre_hook(block_decode)
    async def check():
        worker = GenerationWorker(lambda: (model, Tokenizer()), cache_backend='paged')
        await worker.start()
        task = None
        resume = asyncio.Event()
        try:
            handle = await worker.submit(payload(budget=3))
            sending = asyncio.Event()
            bodies = []
            async def send(message):
                if message['type'] == 'http.response.body':
                    bodies.append(message['body'])
                    if len(bodies) == 1:
                        sending.set()
                        await resume.wait()
            async def receive():
                await asyncio.Event().wait()
            task = asyncio.create_task(GenerationResponse(worker, handle)(
                dict(type='http', asgi={'spec_version': spec_version}), receive, send))
            await asyncio.wait_for(sending.wait(), 5)
            if completed:
                async with asyncio.timeout(5):
                    while not handle.terminal:
                        await asyncio.sleep(0)
                await worker.stop()
            else:
                assert await asyncio.to_thread(entered.wait, 5)
                stopping = asyncio.create_task(worker.stop())
                await asyncio.sleep(0)
                unblock.set()
                await asyncio.wait_for(stopping, 5)
            resume.set()
            await asyncio.wait_for(task, 1)
            events = decode_events(b''.join(bodies).decode())
            assert events[-1][0] == ('done' if completed else 'error')
            assert sum(kind in ('done', 'error') for kind, _ in events) == 1
            assert not worker._handles and not worker.thread.is_alive()
            assert worker.scheduler.page_pool.owned_pages == 0
        finally:
            unblock.set()
            resume.set()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await worker.stop()
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()


@pytest.mark.parametrize('cancellations', [1, 2])
def test_admission_handoff_cancellation_disposes_accepted_handle(cancellations):
    import asyncio
    from server.app import _admit
    from server.worker import GenerationWorker
    from test_server_worker import payload
    async def check():
        worker = GenerationWorker(lambda: (tiny_model(), Tokenizer()), max_outstanding=1)
        await worker.start()
        accepted = []
        class Request:
            async def receive(self):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    accepted.extend(worker._handles.values())
                    for _ in range(cancellations):
                        asyncio.get_running_loop().call_soon(admitting.cancel)
                    raise
        try:
            admitting = asyncio.create_task(_admit(worker, Request(), payload(budget=0)))
            with pytest.raises(asyncio.CancelledError):
                await admitting
            handle, = accepted
            assert handle.cancelled.is_set() and handle.cleanup is not None
            await asyncio.wait_for(asyncio.shield(handle.cleanup), 5)
            assert not worker._handles and not worker.scheduler._requests
        finally:
            await worker.stop()
    asyncio.run(check())
