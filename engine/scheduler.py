"""Synchronous FIFO continuous batching with real-token private KV caches."""
from collections import deque
from dataclasses import dataclass

import torch

from engine.kv_cache import SimpleKVCache
from engine.model import GPT2Model
from engine.sampler import SamplingParams, sample


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int | None
    finish_reason: str | None


@dataclass
class _Request:
    request_id: str
    prompt: torch.Tensor
    output: torch.Tensor
    max_new_tokens: int
    stop_token_ids: frozenset[int]
    sampling: SamplingParams
    generator: torch.Generator
    cache: SimpleKVCache | None = None
    finish_reason: str | None = None


class Scheduler:
    """One owner calls submit/step; finished IDs/results live for this instance.

    Prefill and decode are separate commit phases. Earlier successful events
    survive a later forward failure and are delivered on the next successful step.
    """

    def __init__(self, model: GPT2Model, *, max_batch_size: int, pad_token_id: int) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError('max_batch_size must be a positive integer')
        if type(pad_token_id) is not int or not 0 <= pad_token_id < model.config.vocab_size:
            raise ValueError('pad_token_id must be an integer inside the vocabulary')
        self.model = model
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id
        # ponytail: retain completed results; add eviction when the server owns lifetimes.
        self._requests: dict[str, _Request] = {}
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._pending_events: list[TokenEvent] = []
        self.peak_kv_bytes = 0

    @property
    def idle(self) -> bool:
        return not (self._waiting or self._running or self._pending_events)

    @property
    def cache_allocated_bytes(self) -> int:
        return sum(r.cache.allocated_bytes for r in self._requests.values() if r.cache is not None)

    def submit(self, request_id: str, prompt: torch.Tensor, max_new_tokens: int,
               *, stop_token_ids: tuple[int, ...] = (), sampling: SamplingParams | None = None) -> None:
        device = self.model.token_embedding.weight.device
        if not isinstance(request_id, str) or not request_id or request_id in self._requests:
            raise ValueError('request_id must be a unique nonempty string')
        if not isinstance(prompt, torch.Tensor) or prompt.ndim != 1 or prompt.device != device:
            raise ValueError('prompt must be a rank-1 tensor on the model device')
        self.model.validate_input_ids(prompt[None])
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer')
        if len(prompt) + max_new_tokens > self.model.config.max_positions:
            raise ValueError('Prompt plus output budget exceeds model context')
        if (not isinstance(stop_token_ids, (tuple, list, set, frozenset))
                or any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in stop_token_ids)):
            raise ValueError('stop_token_ids must contain integer vocabulary IDs')
        sampling = SamplingParams() if sampling is None else sampling
        if not isinstance(sampling, SamplingParams):
            raise ValueError('sampling must be SamplingParams')
        sampling.validate(self.model.config.vocab_size)
        accepted = prompt.clone()
        generator = torch.Generator(device=device).manual_seed(sampling.seed)
        self._requests[request_id] = _Request(request_id, accepted, accepted, max_new_tokens,
            frozenset(stop_token_ids), sampling, generator)
        self._waiting.append(request_id)

    def result(self, request_id: str) -> torch.Tensor:
        request = self._requests[request_id]
        if request.finish_reason is None:
            raise ValueError('Request has not finished')
        return request.output.clone()

    def _cache(self, batch_size: int, capacity: int) -> SimpleKVCache:
        weight = self.model.token_embedding.weight
        return SimpleKVCache(self.model.config, batch_size=batch_size, capacity=capacity,
                             device=weight.device, dtype=weight.dtype)

    def _record_peak(self, workspace: SimpleKVCache) -> None:
        self.peak_kv_bytes = max(self.peak_kv_bytes, self.cache_allocated_bytes + workspace.allocated_bytes)

    def _append(self, request: _Request, token: torch.Tensor) -> None:
        token_id = token.item()
        request.output = torch.cat((request.output, token.reshape(1)))
        if token_id in request.stop_token_ids:
            request.finish_reason = 'stop'
        elif len(request.output) - len(request.prompt) == request.max_new_tokens:
            request.finish_reason = 'length'
        if request.finish_reason is not None:
            request.cache = None
            if request.request_id in self._running:
                self._running.remove(request.request_id)
        elif request.request_id not in self._running:
            self._running.append(request.request_id)
        self._pending_events.append(TokenEvent(request.request_id, token_id, request.finish_reason))

    def _prefill(self, requests: list[_Request]) -> None:
        positive = [r for r in requests if r.max_new_tokens]
        workspace = None
        tokens = {}
        if positive:
            width = max(len(r.prompt) for r in positive)
            device = self.model.token_embedding.weight.device
            ids = torch.full((len(positive), width), self.pad_token_id, device=device, dtype=torch.long)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for row, request in enumerate(positive):
                ids[row, -len(request.prompt):] = request.prompt
                mask[row, -len(request.prompt):] = True
            workspace = self._cache(len(positive), width)
            self._record_peak(workspace)
            logits = self.model(ids, cache=workspace, attention_mask=mask)
            for row, request in enumerate(positive):
                tokens[request.request_id] = sample(logits[row, -1], request.sampling, generator=request.generator)
            del logits
        rows = {r.request_id: row for row, r in enumerate(positive)}
        for request in requests:
            if request.max_new_tokens == 0:
                request.finish_reason = 'length'
                self._pending_events.append(TokenEvent(request.request_id, None, 'length'))
                continue
            token = tokens[request.request_id]
            if request.max_new_tokens > 1 and token.item() not in request.stop_token_ids:
                cache = self._cache(1, len(request.prompt) + request.max_new_tokens)
                row = rows[request.request_id]
                length = len(request.prompt)
                cache.keys[:, :, :, :length].copy_(workspace.keys[:, row:row+1, :, width-length:width])
                cache.values[:, :, :, :length].copy_(workspace.values[:, row:row+1, :, width-length:width])
                cache.length = length
                request.cache = cache
                self._record_peak(workspace)
            self._append(request, token)

    def _decode(self, requests: list[_Request]) -> None:
        if not requests:
            return
        past = max(r.cache.length for r in requests)
        workspace = self._cache(len(requests), past + 1)
        # Zero-weight attention still propagates NaNs from uninitialized V slots.
        workspace.keys.zero_()
        workspace.values.zero_()
        workspace.length = past
        device = self.model.token_embedding.weight.device
        mask = torch.zeros((len(requests), past+1), dtype=torch.bool, device=device)
        for row, request in enumerate(requests):
            length = request.cache.length
            workspace.keys[:, row:row+1, :, past-length:past].copy_(request.cache.keys[:, :, :, :length])
            workspace.values[:, row:row+1, :, past-length:past].copy_(request.cache.values[:, :, :, :length])
            mask[row, past-length:] = True
        self._record_peak(workspace)
        ids = torch.stack([r.output[-1] for r in requests])[:, None]
        logits = self.model(ids, cache=workspace, attention_mask=mask)
        tokens = [sample(logits[row, -1], r.sampling, generator=r.generator) for row, r in enumerate(requests)]
        del logits
        for row, (request, token) in enumerate(zip(requests, tokens)):
            cache = request.cache
            length = cache.length
            cache.keys[:, :, :, length:length+1].copy_(workspace.keys[:, row:row+1, :, past:past+1])
            cache.values[:, :, :, length:length+1].copy_(workspace.values[:, row:row+1, :, past:past+1])
            cache.length += 1
            self._append(request, token)

    @torch.inference_mode()
    def step(self) -> list[TokenEvent]:
        old = [self._requests[name] for name in self._running]
        free = self.max_batch_size - len(old)
        cohort = []
        positive = 0
        for name in self._waiting:
            request = self._requests[name]
            if request.max_new_tokens:
                if positive == free:
                    break
                positive += 1
            cohort.append(request)
        if cohort:
            self._prefill(cohort)
            for _ in cohort:
                self._waiting.popleft()
        # ponytail: pack/copy prefixes per step; a shared page pool comes in M5.
        self._decode(old)
        events = self._pending_events
        self._pending_events = []
        return events
