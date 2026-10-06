"""Fixed-capacity contiguous key/value storage owned by one inference request."""
import heapq

import torch

from engine.config import ModelConfig


class SimpleKVCache:
    """Layer tensors use [layers, batch, heads, capacity, head_dim].

    Only positions below length are committed. Writes beyond that prefix are
    tentative until the entire model forward succeeds; retries overwrite them.
    This is inference state, not a differentiable training cache.
    """

    def __init__(self, config: ModelConfig, *, batch_size: int, capacity: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        if (not isinstance(batch_size, int) or batch_size <= 0
                or not isinstance(capacity, int) or not 0 < capacity <= config.max_positions):
            raise ValueError("Cache batch size must be positive; capacity must fit model context")
        if not dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        self.config = config
        self.capacity = capacity
        self.length = 0
        self.requires_attention_mask = False
        shape = (config.num_layers, batch_size, config.num_heads, capacity,
                 config.hidden_size // config.num_heads)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)

    @property
    def allocated_bytes(self) -> int:
        return (self.keys.numel() * self.keys.element_size()
                + self.values.numel() * self.values.element_size())

    @property
    def used_bytes(self) -> int:
        return self.allocated_bytes // self.capacity * self.length

    def validate(self, config: ModelConfig, input_ids: torch.Tensor, *,
                 device: torch.device, dtype: torch.dtype) -> None:
        if self.config != config or input_ids.shape[0] != self.keys.shape[1]:
            raise ValueError("Cache model dimensions or request batch do not match")
        if (self.keys.device != device or input_ids.device != device
                or self.keys.dtype != dtype):
            raise ValueError("Cache, model, and request must have matching device/dtype")
        if not 0 <= self.length <= self.capacity or self.length + input_ids.shape[1] > self.capacity:
            raise ValueError("Request exceeds KV cache capacity")

    @torch.no_grad()
    def write(self, layer_idx: int, key: torch.Tensor,
              value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.config.num_layers:
            raise ValueError("Invalid cache layer index")
        expected = (self.keys.shape[1], self.config.num_heads, key.shape[-2],
                    self.config.hidden_size // self.config.num_heads)
        if (key.shape != expected or value.shape != expected or key.shape[-2] == 0
                or key.dtype != self.keys.dtype or value.dtype != self.values.dtype
                or key.device != self.keys.device or value.device != self.values.device):
            raise ValueError("Invalid key/value chunk shape, dtype, or device")
        end = self.length + key.shape[-2]
        if end > self.capacity:
            raise ValueError("Request exceeds KV cache capacity")
        self.keys[layer_idx, :, :, self.length:end].copy_(key)
        self.values[layer_idx, :, :, self.length:end].copy_(value)
        return self.keys[layer_idx, :, :, :end], self.values[layer_idx, :, :, :end]


class PagePool:
    """Fixed resident K/V storage; a page ID spans every transformer layer."""

    def __init__(self, config: ModelConfig, *, num_pages: int, page_size: int = 16,
                 device: torch.device, dtype: torch.dtype) -> None:
        if (not isinstance(config, ModelConfig) or type(num_pages) is not int or num_pages <= 0
                or type(page_size) is not int or page_size <= 0):
            raise ValueError('Pool requires model dimensions and positive integer page counts/size')
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError('KV pool dtype must be floating point')
        self.config = config
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = torch.device(device)
        self.dtype = dtype
        shape = (config.num_layers, num_pages, config.num_heads, page_size,
                 config.hidden_size // config.num_heads)
        self.keys = torch.empty(shape, device=self.device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self._free = list(range(num_pages))
        self._owned: set[int] = set()

    @property
    def allocated_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values))

    @property
    def page_bytes(self) -> int:
        return self.allocated_bytes // self.num_pages

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def owned_pages(self) -> int:
        return len(self._owned)

    @property
    def free_page_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._free))

    def allocate(self, count: int) -> tuple[int, ...]:
        if type(count) is not int or count <= 0:
            raise ValueError('Page allocation count must be a positive integer')
        if count > self.free_pages:
            raise MemoryError('KV page pool capacity exhausted')
        ids = tuple(heapq.heappop(self._free) for _ in range(count))
        self._owned.update(ids)
        return ids

    def release(self, page_ids: tuple[int, ...]) -> None:
        if (not isinstance(page_ids, (tuple, list))
                or any(type(i) is not int or i not in self._owned for i in page_ids)
                or len(set(page_ids)) != len(page_ids)):
            raise ValueError('Returned page IDs must be unique and currently allocated')
        for i in page_ids:
            self._owned.remove(i)
            heapq.heappush(self._free, i)
