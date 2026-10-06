"""Hugging Face loading boundary; the engine executes only its own model."""
from collections.abc import Mapping
import os
from pathlib import Path

import torch

# Keep Hugging Face's auxiliary caches local too; never require a user's home cache.
os.environ.setdefault("HF_HOME", str(Path("model_cache").resolve()))
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase

from engine.config import EngineConfig, ModelConfig
from engine.model import GPT2Model


def resolve_device(device: str) -> torch.device:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(device)


def copy_hf_weights(model: GPT2Model, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Validate the whole mapping before copying, including square Conv1D weights."""
    mapping: dict[str, tuple[str, bool]] = {
        "token_embedding.weight": ("transformer.wte.weight", False),
        "position_embedding.weight": ("transformer.wpe.weight", False),
        "final_norm.weight": ("transformer.ln_f.weight", False),
        "final_norm.bias": ("transformer.ln_f.bias", False),
    }
    block_names = {
        "attention_norm": "ln_1", "attention.qkv": "attn.c_attn",
        "attention.projection": "attn.c_proj", "mlp_norm": "ln_2",
        "mlp.up": "mlp.c_fc", "mlp.down": "mlp.c_proj",
    }
    for i in range(model.config.num_layers):
        for own, hf in block_names.items():
            for suffix in ("weight", "bias"):
                transpose = suffix == "weight" and ".c_" in hf
                mapping[f"blocks.{i}.{own}.{suffix}"] = (
                    f"transformer.h.{i}.{hf}.{suffix}", transpose)
    copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        key, transpose = mapping[name]
        if key not in state_dict:
            expected_shape = tuple(parameter.T.shape if transpose else parameter.shape)
            raise ValueError(f"Missing checkpoint tensor: {key}; expected shape {expected_shape}")
        source = state_dict[key]
        if transpose:
            source = source.T
        if source.shape != parameter.shape:
            raise ValueError(f"Checkpoint tensor {key}: expected mapped shape "
                             f"{tuple(parameter.shape)}, got {tuple(source.shape)}")
        copies.append((parameter, source))
    with torch.no_grad():
        for parameter, source in copies:
            parameter.copy_(source)


def load_model(config: EngineConfig) -> tuple[GPT2Model, PreTrainedTokenizerBase]:
    """Load public GPT-2 weights in FP32 without retaining the HF reference model."""
    device = resolve_device(config.device)
    hf_config = GPT2Config.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    if (hf_config.model_type != "gpt2" or hf_config.add_cross_attention
            or not hf_config.scale_attn_weights or hf_config.scale_attn_by_inverse_layer_idx
            or hf_config.reorder_and_upcast_attn):
        raise ValueError("Only standard GPT-2 self-attention configuration is supported")
    model_config = ModelConfig(
        vocab_size=hf_config.vocab_size, max_positions=hf_config.n_positions,
        hidden_size=hf_config.n_embd, num_layers=hf_config.n_layer,
        num_heads=hf_config.n_head, intermediate_size=hf_config.n_inner or 4 * hf_config.n_embd,
        layer_norm_epsilon=hf_config.layer_norm_epsilon,
        activation_function=hf_config.activation_function)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    reference, loading_info = GPT2LMHeadModel.from_pretrained(
        config.model_name, config=hf_config, cache_dir=config.cache_dir,
        torch_dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
    # HF fills absent parameters with random values. Inspect its diagnostics
    # before state_dict() makes those synthesized values look like real weights.
    if loading_info["missing_keys"]:
        shapes = reference.state_dict()
        missing = "; ".join(f"{key}: expected shape {tuple(shapes[key].shape)}"
                            for key in loading_info["missing_keys"])
        raise ValueError(f"Incomplete checkpoint: {missing}")
    if loading_info["mismatched_keys"] or loading_info["error_msgs"]:
        raise ValueError(f"Invalid checkpoint: {loading_info}")
    model = GPT2Model(model_config)
    copy_hf_weights(model, reference.state_dict())
    del reference
    if config.int8:
        from engine.quantize import quantize_model
        quantize_model(model)
    return model.to(device).eval(), tokenizer
