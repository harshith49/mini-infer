"""Synchronous FIFO continuous batching with real-token private KV caches."""
import argparse
from collections import deque
import json
from dataclasses import dataclass

import torch

from engine.config import EngineConfig, ModelConfig
from engine.kv_cache import PagePool, PagedKVCache, SimpleKVCache
from engine.model import GPT2Model
from engine.sampler import SamplingParams, sample
from engine.weights import load_model


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
    cache: SimpleKVCache | PagedKVCache | None = None
    finish_reason: str | None = None


class Scheduler:
    """One owner calls submit/step; finished IDs/results live for this instance.

    Prefill and decode are separate commit phases. Earlier successful events
    survive a later forward failure and are delivered on the next successful step.
    """

    def __init__(self, model: GPT2Model, *, max_batch_size: int, pad_token_id: int,
                 page_pool: PagePool | None = None) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError('max_batch_size must be a positive integer')
        if type(pad_token_id) is not int or not 0 <= pad_token_id < model.config.vocab_size:
            raise ValueError('pad_token_id must be an integer inside the vocabulary')
        weight = model.token_embedding.weight
        if page_pool is not None and (not isinstance(page_pool, PagePool)
                or page_pool.config != model.config or page_pool.device != weight.device
                or page_pool.dtype != weight.dtype or page_pool.free_pages != page_pool.num_pages):
            raise ValueError('Scheduler requires a matching entirely free page pool')
        self.page_pool = page_pool
        self.model = model
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id
        # ponytail: retain completed results; add eviction when the server owns lifetimes.
        self._requests: dict[str, _Request] = {}
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._pending_events: list[TokenEvent] = []
        self.peak_kv_bytes = self.pool_resident_bytes

    @property
    def idle(self) -> bool:
        return not (self._waiting or self._running or self._pending_events)

    @property
    def pool_resident_bytes(self) -> int:
        return self.page_pool.allocated_bytes if self.page_pool is not None else 0

    @property
    def cache_allocated_bytes(self) -> int:
        if self.page_pool is not None:
            return self.page_pool.owned_pages * self.page_pool.page_bytes
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
        if (self.page_pool is not None and max_new_tokens
                and self._required_pages(len(prompt) + max_new_tokens) > self.page_pool.num_pages):
            raise ValueError('Request full budget exceeds the entire KV page pool')
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

    def _required_pages(self, capacity: int) -> int:
        return (capacity + self.page_pool.page_size - 1) // self.page_pool.page_size

    def _record_peak(self, workspace: SimpleKVCache, *, gather_bytes: int = 0) -> None:
        resident = self.pool_resident_bytes if self.page_pool is not None else self.cache_allocated_bytes
        self.peak_kv_bytes = max(self.peak_kv_bytes, resident + workspace.allocated_bytes + gather_bytes)

    def _append(self, request: _Request, token: torch.Tensor) -> None:
        token_id = token.item()
        request.output = torch.cat((request.output, token.reshape(1)))
        if token_id in request.stop_token_ids:
            request.finish_reason = 'stop'
        elif len(request.output) - len(request.prompt) == request.max_new_tokens:
            request.finish_reason = 'length'
        if request.finish_reason is not None:
            if isinstance(request.cache, PagedKVCache):
                request.cache.close()
            request.cache = None
            if request.request_id in self._running:
                self._running.remove(request.request_id)
        elif request.request_id not in self._running:
            self._running.append(request.request_id)
        self._pending_events.append(TokenEvent(request.request_id, token_id, request.finish_reason))

    def _prefill(self, requests: list[_Request]) -> None:
        staged: dict[str, PagedKVCache] = {}
        try:
            if self.page_pool is not None:
                for request in requests:
                    if request.max_new_tokens:
                        staged[request.request_id] = PagedKVCache(self.page_pool,
                            capacity=len(request.prompt) + request.max_new_tokens)
            self._prefill_reserved(requests, staged)
        finally:
            # Attached continuing caches leave this dict. Failed admissions and
            # first-token completions return every remaining reservation.
            for cache in staged.values():
                cache.close()

    def _prefill_reserved(self, requests: list[_Request], staged: dict[str, PagedKVCache]) -> None:
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
                cache = (staged[request.request_id] if self.page_pool is not None
                         else self._cache(1, len(request.prompt) + request.max_new_tokens))
                row = rows[request.request_id]
                length = len(request.prompt)
                for layer in range(self.model.config.num_layers):
                    cache.store(layer, 0, workspace.keys[layer, row:row+1, :, width-length:width],
                                workspace.values[layer, row:row+1, :, width-length:width])
                cache.length = length
                request.cache = cache
                staged.pop(request.request_id, None)
                self._record_peak(workspace)
            elif request.request_id in staged:
                staged.pop(request.request_id).close()
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
            for layer in range(self.model.config.num_layers):
                keys, values = request.cache.read_prefix(layer)
                gather_bytes = (sum(t.numel() * t.element_size() for t in (keys, values))
                                if isinstance(request.cache, PagedKVCache) else 0)
                self._record_peak(workspace, gather_bytes=gather_bytes)
                workspace.keys[layer, row:row+1, :, past-length:past].copy_(keys)
                workspace.values[layer, row:row+1, :, past-length:past].copy_(values)
                del keys, values
            mask[row, past-length:] = True
        self._record_peak(workspace)
        ids = torch.stack([r.output[-1] for r in requests])[:, None]
        logits = self.model(ids, cache=workspace, attention_mask=mask)
        tokens = [sample(logits[row, -1], r.sampling, generator=r.generator) for row, r in enumerate(requests)]
        del logits
        for row, (request, token) in enumerate(zip(requests, tokens)):
            cache = request.cache
            length = cache.length
            for layer in range(self.model.config.num_layers):
                cache.store(layer, length, workspace.keys[layer, row:row+1, :, past:past+1],
                            workspace.values[layer, row:row+1, :, past:past+1])
            cache.length += 1
            self._append(request, token)

    @torch.inference_mode()
    def step(self) -> list[TokenEvent]:
        old = [self._requests[name] for name in self._running]
        free = self.max_batch_size - len(old)
        cohort = []
        positive = 0
        pages_available = self.page_pool.free_pages if self.page_pool is not None else None
        for name in self._waiting:
            request = self._requests[name]
            if request.max_new_tokens:
                if positive == free:
                    break
                if pages_available is not None:
                    needed = self._required_pages(len(request.prompt) + request.max_new_tokens)
                    if needed > pages_available:
                        break
                    pages_available -= needed
                positive += 1
            cohort.append(request)
        if cohort:
            self._prefill(cohort)
            for _ in cohort:
                self._waiting.popleft()
        # ponytail: gather/copy prefixes per step; custom paged attention kernels remain later work.
        self._decode(old)
        events = self._pending_events
        self._pending_events = []
        return events


def main() -> None:
    parser = argparse.ArgumentParser(description='mini-infer: FIFO continuous batching')
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--max-new-tokens', nargs='+', type=int, default=[50])
    parser.add_argument('--max-batch-size', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-backend', choices=('contiguous', 'paged'), default='contiguous')
    parser.add_argument('--num-pages', type=int, default=32)
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument("--int8", action="store_true", help="Store transformer projection weights as int8")
    args = parser.parse_args()
    try:
        if len(args.max_new_tokens) not in (1, len(args.prompt)) or any(n < 0 for n in args.max_new_tokens):
            raise ValueError('Use one nonnegative output budget or one per prompt')
        if args.max_batch_size <= 0 or args.num_pages <= 0 or args.page_size <= 0:
            raise ValueError('max_batch_size, num_pages and page_size must be positive')
        settings = SamplingParams(args.temperature, args.top_k, args.top_p, args.seed)
        # This CLI loads public GPT-2; validate its default vocabulary before download.
        settings.validate(ModelConfig().vocab_size)
        model, tokenizer = load_model(EngineConfig(device=args.device, int8=args.int8))
        device = model.token_embedding.weight.device
        pool = (PagePool(model.config, num_pages=args.num_pages, page_size=args.page_size,
            device=device, dtype=model.token_embedding.weight.dtype) if args.cache_backend == 'paged' else None)
        scheduler = Scheduler(model, max_batch_size=args.max_batch_size,
                              pad_token_id=tokenizer.eos_token_id, page_pool=pool)
        prompts = []
        for i, text in enumerate(args.prompt):
            prompt = (tokenizer(text, return_tensors='pt')['input_ids'][0] if text
                      else torch.tensor([tokenizer.eos_token_id], dtype=torch.long)).to(device)
            budget = args.max_new_tokens[0] if len(args.max_new_tokens) == 1 else args.max_new_tokens[i]
            scheduler.submit(str(i), prompt, budget, stop_token_ids=(tokenizer.eos_token_id,), sampling=settings)
            prompts.append(prompt)
        while not scheduler.idle:
            scheduler.step()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps([text + tokenizer.decode(scheduler.result(str(i))[len(prompt):].tolist(), skip_special_tokens=True)
        for i, (text, prompt) in enumerate(zip(args.prompt, prompts))], ensure_ascii=False))


if __name__ == '__main__':
    main()
