"""Fixed-capacity contiguous key/value storage owned by one inference request."""
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
