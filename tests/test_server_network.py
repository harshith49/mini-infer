"""Actual TCP streaming, overlapping batching, disconnect and shutdown."""
import asyncio
from contextlib import asynccontextmanager
import json
import socket
import threading

import httpx
import pytest
import uvicorn

from test_server_worker import tiny_model, Tokenizer


@asynccontextmanager
async def running(app):
    from server.app import ServingServer
    sock = socket.socket()
    sock.bind(('127.0.0.1',0))
    port = sock.getsockname()[1]
    server = ServingServer(uvicorn.Config(app, log_level='error', lifespan='on'))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(.01)  # Condition polling, no assumed startup duration.
        yield server, f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        await asyncio.wait_for(task,10)
        sock.close()


async def next_event(lines):
    kind = None
    data = None
    async with asyncio.timeout(10):
        async for line in lines:
            if line.startswith('event: '):
                kind = line[7:]
            elif line.startswith('data: '):
                data = json.loads(line[6:])
            elif not line and kind:
                return kind,data
    raise AssertionError('premature SSE EOF')


def test_real_stream_arrives_before_completion_and_batches():
    from server.app import create_app
    model = tiny_model()
    blocked, unblock = threading.Event(), threading.Event()
    batches = []
    calls = 0
    def forward(m,a):
        nonlocal calls
        calls += 1
        batches.append(len(a[0]))
        if calls == 2:
            blocked.set()
            assert unblock.wait(10)
    hook = model.register_forward_pre_hook(forward)
    async def check():
        app = create_app(loader=lambda: (model,Tokenizer()),max_batch_size=2)
        async with running(app) as (_,url), httpx.AsyncClient(timeout=10) as c:
            async with c.stream('POST',url+'/v1/generate',json=dict(prompt='A',max_new_tokens=6,stop_token_ids=[])) as first:
                lines = first.aiter_lines()
                kind,data = await next_event(lines)
                assert kind == 'token'
                assert await asyncio.to_thread(blocked.wait,5)
                # HTTP handler can enqueue this request while inference is blocked.
                second = asyncio.create_task(c.post(url+'/v1/generate',json=dict(prompt='B',max_new_tokens=4,stop_token_ids=[])))
                async with asyncio.timeout(5):
                    while len(app.state.worker._handles) < 2:
                        await asyncio.sleep(.01)
                unblock.set()
                rest = []
                while kind != 'done':
                    kind,data = await next_event(lines)
                    rest.append(kind)
                assert rest[-1] == 'done'
                response = await second
                assert response.status_code == 200 and 'event: done' in response.text
                assert max(batches) == 2
        assert not app.state.worker.thread.is_alive()
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()


def test_real_disconnect_releases_pages_without_harming_peer():
    from server.app import create_app
    model = tiny_model()
    entered, unblock = threading.Event(), threading.Event()
    calls = 0
    def block(m,a):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert unblock.wait(10)
    hook=model.register_forward_pre_hook(block)
    async def check():
        app=create_app(loader=lambda:(model,Tokenizer()),cache_backend='paged',num_pages=16,page_size=4)
        async with running(app) as (_,url), httpx.AsyncClient(timeout=10) as c:
            async with c.stream('POST',url+'/v1/generate',json=dict(prompt='A',max_new_tokens=8,stop_token_ids=[])) as response:
                kind,data=await next_event(response.aiter_lines())
                assert kind=='token'
                assert await asyncio.to_thread(entered.wait,5)
                handle=app.state.worker._handles[data['request_id']]
            # Disconnect reaches response cancellation before backend release.
            async with asyncio.timeout(5):
                while not handle.cancelled.is_set():
                    await asyncio.sleep(.01)
            peer=asyncio.create_task(c.post(url+'/v1/generate',json=dict(prompt='B',max_new_tokens=3,stop_token_ids=[])))
            unblock.set()
            await asyncio.wait_for(app.state.worker.release(handle),5)
            result=await peer
            assert result.status_code==200 and 'event: done' in result.text
            assert handle.request_id not in app.state.worker.scheduler._requests
        assert app.state.worker.scheduler.page_pool.owned_pages==0
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()


def test_real_capacity_fault_and_graceful_shutdown():
    from server.app import create_app
    model=tiny_model()
    entered,unblock=threading.Event(),threading.Event()
    def fail(m,a):
        entered.set()
        assert unblock.wait(10)
        raise RuntimeError('secret backend detail')
    hook=model.register_forward_pre_hook(fail)
    async def check():
        app=create_app(loader=lambda:(model,Tokenizer()),max_outstanding=1,cache_backend='paged')
        async with running(app) as (_,url),httpx.AsyncClient(timeout=10) as c:
            first=asyncio.create_task(c.post(url+'/v1/generate',json=dict(prompt='A',max_new_tokens=3)))
            assert await asyncio.to_thread(entered.wait,5)
            assert (await c.post(url+'/v1/generate',json=dict(prompt='B'))).status_code==429
            unblock.set()
            response=await first
            assert 'event: error' in response.text and 'event: done' not in response.text
            assert 'secret backend detail' not in response.text
            assert (await c.post(url+'/v1/generate',json=dict(prompt='B'))).status_code==503
        assert not app.state.worker._handles and app.state.worker.scheduler.page_pool.owned_pages==0
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()


def test_real_cancel_before_headers_and_shutdown_active_stream():
    from server.app import create_app
    model=tiny_model()
    entered,unblock=threading.Event(),threading.Event()
    calls=0
    def block(m,a):
        nonlocal calls
        calls+=1
        if calls==2:
            entered.set()
            assert unblock.wait(10)
    hook=model.register_forward_pre_hook(block)
    async def check():
        app=create_app(loader=lambda:(model,Tokenizer()),max_outstanding=3,cache_backend='paged')
        async with running(app) as (server,url),httpx.AsyncClient(timeout=10) as c:
            async with c.stream('POST',url+'/v1/generate',json=dict(prompt='A',max_new_tokens=8,stop_token_ids=[])) as first:
                lines=first.aiter_lines()
                assert (await next_event(lines))[0]=='token'
                assert await asyncio.to_thread(entered.wait,5)
                pending=asyncio.create_task(c.post(url+'/v1/generate',json=dict(prompt='B',max_new_tokens=4)))
                async with asyncio.timeout(5):
                    while len(app.state.worker._handles)<2:
                        await asyncio.sleep(.01)
                abandoned=next(h for h in app.state.worker._handles.values() if not h.admitted.done())
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                async with asyncio.timeout(5):
                    while not abandoned.cancelled.is_set():
                        await asyncio.sleep(.01)
                server.should_exit=True
                async with asyncio.timeout(5):
                    while app.state.worker._state!='stopping':
                        await asyncio.sleep(.01)
                unblock.set()
                terminal=None
                while terminal not in ('done','error'):
                    terminal,_=await next_event(lines)
                assert terminal=='error'
        assert not app.state.worker._handles and app.state.worker.scheduler.page_pool.owned_pages==0
        assert not app.state.worker.thread.is_alive()
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()
