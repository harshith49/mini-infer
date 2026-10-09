"""Strict generation API. Run with python -m server.app for coordinated shutdown."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import socket
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch
import uvicorn

from engine.config import EngineConfig
from engine.model import GPT2Model
from engine.weights import load_model
from server.worker import AdmissionError, GenerationWorker, StreamHandle


class GenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)
    prompt: str = Field(max_length=65536)
    max_new_tokens: int = Field(default=50, ge=0, le=1024)
    temperature: float = Field(default=0., ge=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1., gt=0, le=1)
    seed: int = Field(default=0, ge=0, lt=2**63)
    stop_token_ids: list[int] | None = None

    @field_validator('stop_token_ids', mode='before')
    @classmethod
    def stops_are_a_list(cls, value):
        if not isinstance(value, list):
            raise ValueError('stop_token_ids must be a list of integer vocabulary IDs')
        return value


def encode_sse(event: str, data: dict[str, object]) -> bytes:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=True, allow_nan=False)}\n\n'.encode()


class GenerationResponse(StreamingResponse):
    """Dispose even when headers/send fail before the generator can run."""
    def __init__(self, worker, handle):
        self.worker, self.handle = worker, handle
        async def stream():
            while True:
                event, data = await handle.events.get()
                yield encode_sse(event, data)
                if event in ('done', 'error'):
                    return
        super().__init__(stream(), media_type='text/event-stream', headers={'Cache-Control':'no-cache'})

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.worker.release(self.handle)


async def _admit(worker, request, payload):
    submission = asyncio.create_task(worker.submit(payload))
    async def disconnected():
        while (await request.receive())['type'] != 'http.disconnect':
            pass
    disconnect = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait([submission, disconnect], return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            raise HTTPException(499, 'Client disconnected')
        return await submission
    except BaseException:
        submission.cancel()
        result, = await asyncio.gather(submission, return_exceptions=True)
        if isinstance(result, StreamHandle):
            await worker.release(result)
        raise
    finally:
        disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)


def create_app(*, loader: Callable[[], tuple[GPT2Model, object]] | None = None,
               engine_config: EngineConfig | None = None, max_batch_size: int = 2,
               max_outstanding: int = 64, cache_backend: str = 'contiguous',
               num_pages: int = 32, page_size: int = 16) -> FastAPI:
    config = engine_config or EngineConfig()
    @asynccontextmanager
    async def lifespan(app):
        worker = GenerationWorker(loader or (lambda: load_model(config)),
            max_batch_size=max_batch_size, max_outstanding=max_outstanding,
            cache_backend=cache_backend, num_pages=num_pages, page_size=page_size)
        app.state.worker = worker
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()
    app = FastAPI(title='mini-infer', lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Input NaN/Infinity cannot be serialized into a JSON error response.
        return JSONResponse(status_code=422, content={'detail':[
            {k:e[k] for k in ('loc','msg','type')} for e in error.errors()]})

    @app.post('/v1/generate')
    async def generation(payload: GenerateRequest, request: Request):
        worker = app.state.worker
        try:
            handle = await _admit(worker, request, payload.model_dump())
        except AdmissionError as error:
            raise HTTPException(error.status_code, str(error)) from None
        return GenerationResponse(worker, handle)
    return app


class ServingServer(uvicorn.Server):
    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        worker = getattr(self.config.app.state, 'worker', None)
        if worker is not None:
            await worker.stop()
        await super().shutdown(sockets=sockets)


app = create_app()  # Importing the module never loads a checkpoint.


def main() -> None:
    parser = argparse.ArgumentParser(description='mini-infer: streaming HTTP generation')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--max-batch-size', type=int, default=2)
    parser.add_argument('--max-outstanding', type=int, default=64)
    parser.add_argument('--cache-backend', choices=('contiguous','paged'), default='contiguous')
    parser.add_argument('--num-pages', type=int, default=32)
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or min(args.max_batch_size,args.max_outstanding,
                                        args.num_pages,args.page_size,args.threads) <= 0:
        parser.error('Use port 1..65535 and positive batch/stream/page/thread counts')
    torch.set_num_threads(args.threads)
    selected = create_app(engine_config=EngineConfig(device=args.device,int8=args.int8),
        max_batch_size=args.max_batch_size,max_outstanding=args.max_outstanding,
        cache_backend=args.cache_backend,num_pages=args.num_pages,page_size=args.page_size)
    asyncio.run(ServingServer(uvicorn.Config(selected, host=args.host, port=args.port,
        workers=1, access_log=False)).serve())


if __name__ == '__main__':
    main()
