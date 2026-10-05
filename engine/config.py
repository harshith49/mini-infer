"""Configuration for the baseline GPT-2 architecture and loading boundary."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    max_positions: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    def __post_init__(self) -> None:
        dimensions = (self.vocab_size, self.max_positions, self.hidden_size,
                      self.num_layers, self.num_heads, self.intermediate_size)
        if any(not isinstance(d, int) or d <= 0 for d in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.activation_function != "gelu_new":
            raise ValueError("Only GPT-2 gelu_new activation is supported")


@dataclass(frozen=True)
class EngineConfig:
    device: str = "auto"
    model_name: str = "gpt2"
    cache_dir: str = "model_cache"
