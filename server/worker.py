"""One inference owner; finite event channels and explicit request disposal."""
import asyncio
from dataclasses import dataclass, field
import queue
import threading
from typing import Callable
import uuid

import torch

from engine.kv_cache import PagePool
from engine.model import GPT2Model
from engine.sampler import SamplingParams
from engine.scheduler import Scheduler


class AdmissionError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class TextDeltas:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ''

    def push(self, token_id: int, *, final: bool = False) -> str:
        self.ids.append(token_id)
        # ponytail: quadratic decode is bounded by 1024 tokens; use a byte decoder for larger contexts.
        decoded = self.tokenizer.decode(self.ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        stable = decoded if final else decoded.rstrip('\ufffd')
        if not stable.startswith(self.text):
            raise ValueError('Tokenizer changed the emitted text prefix')
        delta = stable[len(self.text):]
        self.text = stable
        return delta


@dataclass
class StreamHandle:
    request_id: str
    events: asyncio.Queue
    admitted: asyncio.Future
    disposed: asyncio.Future
    cancelled: threading.Event = field(default_factory=threading.Event)
    terminal: bool = False
    cleanup: asyncio.Task | None = None


class GenerationWorker:
    def __init__(self, loader: Callable[[], tuple[GPT2Model, object]], *,
                 max_batch_size: int = 2, max_outstanding: int = 64,
                 cache_backend: str = 'contiguous', num_pages: int = 32, page_size: int = 16):
        if any(type(n) is not int or n <= 0 for n in
               [max_batch_size, max_outstanding, num_pages, page_size]):
            raise ValueError('Worker limits must be positive integers')
        if cache_backend not in ('contiguous', 'paged'):
            raise ValueError('Unknown cache backend')
        self.loader = loader
        self.max_batch_size, self.max_outstanding = max_batch_size, max_outstanding
        self.cache_backend, self.num_pages, self.page_size = cache_backend, num_pages, page_size
        self._commands = queue.Queue()
        self._handles: dict[str, StreamHandle] = {}  # Event-loop owner only.
        self._jobs: dict[str, tuple[dict, int, TextDeltas, StreamHandle]] = {}  # Inference owner only.
        self._lock = threading.Lock()
        self._alive = False
        self._state = 'new'
        self.scheduler = None
        self.thread = None

    async def start(self) -> None:
        if self._state != 'new':
            raise RuntimeError('Worker can only be started once')
        self.loop = asyncio.get_running_loop()
        self.ready = self.loop.create_future()
        self._state = 'starting'
        self._alive = True
        self.thread = threading.Thread(target=self._run, name='mini-infer-worker')
        self.thread.start()
        try:
            await asyncio.shield(self.ready)
            self._state = 'open'
        except BaseException:
            await self.stop()
            raise

    def _enqueue(self, command) -> bool:
        # Coordinate the final queue drain with producers, including a death/admission race.
        with self._lock:
            if not self._alive:
                return False
            self._commands.put(command)
            return True

    @staticmethod
    def _resolve(future, value=None, error=None):
        if not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(value)

    async def submit(self, payload: dict[str, object]) -> StreamHandle:
        if self._state != 'open':
            raise AdmissionError(503, 'Generation worker is unavailable')
        if len(self._handles) >= self.max_outstanding:
            raise AdmissionError(429, 'Outstanding stream capacity exhausted')
        handle = StreamHandle(uuid.uuid4().hex,
            asyncio.Queue(maxsize=max(1, payload['max_new_tokens'] + 1)),
            self.loop.create_future(), self.loop.create_future())
        # Consume abandoned admission exceptions when the HTTP task is cancelled.
        handle.admitted.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self._handles[handle.request_id] = handle
        if not self._enqueue(('submit', handle, payload)):
            self._resolve(handle.admitted, error=AdmissionError(503, 'Generation worker is unavailable'))
        try:
            await asyncio.shield(handle.admitted)
            return handle
        except asyncio.CancelledError:
            self._begin_release(handle)
            raise
        except BaseException:
            await self.release(handle)
            raise

    def _begin_release(self, handle):
        if handle.cleanup is None:
            handle.cancelled.set()
            handle.cleanup = self.loop.create_task(self._release(handle))
        return handle.cleanup

    async def _release(self, handle):
        if not self._enqueue(('cancel', handle, None)):
            self._resolve(handle.disposed)
        await asyncio.shield(handle.disposed)
        self._handles.pop(handle.request_id, None)
        # Discard queued payloads as soon as transport ownership ends.
        while not handle.events.empty():
            handle.events.get_nowait()

    async def release(self, handle: StreamHandle) -> None:
        await asyncio.shield(self._begin_release(handle))

    async def stop(self) -> None:
        if self._state == 'new':
            self._state = 'stopped'
            return
        if self._state not in ('failed', 'stopped'):
            self._state = 'stopping'
            self._enqueue(('stop', None, None))
        if self.thread is not None:
            await asyncio.to_thread(self.thread.join)
        await asyncio.gather(*(self.release(h) for h in list(self._handles.values())))
        self._state = 'stopped'

    def _deliver(self, handle, kind, data):
        if not handle.cancelled.is_set() and not handle.terminal:
            handle.events.put_nowait((kind, data))
            handle.terminal = kind in ('done', 'error')

    def _fail(self, code):
        if self._state != 'stopping':
            self._state = 'failed'
        for handle in self._handles.values():
            if not handle.admitted.done():
                self._resolve(handle.admitted, error=AdmissionError(503, 'Generation worker is unavailable'))
            else:
                self._deliver(handle, 'error', dict(request_id=handle.request_id,
                    code=code, message='Generation stopped; restart the server' if code == 'worker_failed'
                    else 'Server is shutting down'))

    def _submit(self, handle, payload):
        if handle.cancelled.is_set():
            self.loop.call_soon_threadsafe(self._resolve, handle.admitted, None,
                AdmissionError(503, 'Request cancelled'))
            return
        try:
            text = payload['prompt']
            ids = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            if not len(ids):
                ids = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            ids = ids.to(self.scheduler.model.token_embedding.weight.device)
            stops = payload.get('stop_token_ids')
            if stops is None:
                stops = [self.tokenizer.eos_token_id]
            settings = SamplingParams(*(payload[k] for k in ['temperature','top_k','top_p','seed']))
            self.scheduler.submit(handle.request_id, ids, payload['max_new_tokens'],
                                  stop_token_ids=stops, sampling=settings)
        except ValueError as error:
            self.loop.call_soon_threadsafe(self._resolve, handle.admitted, None,
                                          AdmissionError(422, str(error)))
            return
        self._jobs[handle.request_id] = (payload, len(ids), TextDeltas(self.tokenizer), handle)
        self.loop.call_soon_threadsafe(self._resolve, handle.admitted)

    def _command(self, command):
        kind, handle, payload = command
        if kind == 'stop':
            return False
        if kind == 'submit':
            self._submit(handle, payload)
        else:
            self.scheduler.cancel(handle.request_id)
            self._jobs.pop(handle.request_id, None)
            self.loop.call_soon_threadsafe(self._resolve, handle.disposed)
        return True

    def _step(self):
        for event in self.scheduler.step():
            payload, prompt_count, text, handle = self._jobs[event.request_id]
            if event.token_id is not None:
                data = dict(request_id=event.request_id, token_id=event.token_id,
                            delta=text.push(event.token_id, final=event.finish_reason is not None))
                self.loop.call_soon_threadsafe(self._deliver, handle, 'token', data)
            if event.finish_reason is not None:
                data = dict(request_id=event.request_id, finish_reason=event.finish_reason,
                    token_ids=list(text.ids), text=payload['prompt'] + text.text,
                    usage=dict(prompt_tokens=prompt_count, completion_tokens=len(text.ids),
                               total_tokens=prompt_count + len(text.ids)),
                    server_peak_forward_batch_size=self.scheduler.peak_forward_batch_size)
                self.scheduler.discard(event.request_id)
                self._jobs.pop(event.request_id)
                self.loop.call_soon_threadsafe(self._deliver, handle, 'done', data)

    def _run(self):
        code = 'server_shutdown'
        try:
            model, self.tokenizer = self.loader()
            weight = model.token_embedding.weight
            pool = (PagePool(model.config, num_pages=self.num_pages, page_size=self.page_size,
                device=weight.device, dtype=weight.dtype) if self.cache_backend == 'paged' else None)
            self.scheduler = Scheduler(model, max_batch_size=self.max_batch_size,
                                       pad_token_id=self.tokenizer.eos_token_id, page_pool=pool)
            self.loop.call_soon_threadsafe(self._resolve, self.ready)
            running = True
            while running:
                if self.scheduler.idle:
                    running = self._command(self._commands.get())
                while running:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    running = self._command(command)
                if running and not self.scheduler.idle:
                    self._step()
        except BaseException as error:
            code = 'worker_failed'
            self.loop.call_soon_threadsafe(self._resolve, self.ready, None, error)
        finally:
            with self._lock:
                self._alive = False
            if self.scheduler is not None:
                self.scheduler.close()
            self._jobs.clear()
            self.loop.call_soon_threadsafe(self._fail, code)
            while True:
                try:
                    kind, handle, _ = self._commands.get_nowait()
                except queue.Empty:
                    break
                if kind == 'cancel':
                    self.loop.call_soon_threadsafe(self._resolve, handle.disposed)
                elif kind == 'submit':
                    self.loop.call_soon_threadsafe(self._resolve, handle.admitted, None,
                                                  AdmissionError(503, 'Generation worker is unavailable'))
