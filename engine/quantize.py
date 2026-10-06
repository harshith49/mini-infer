"""Transparent inference-only int8 storage; floating matmul stays in PyTorch."""
import torch
from torch import nn
from torch.nn import functional as F

from engine.model import GPT2Model


class Int8Linear(nn.Module):
    """One symmetric scale per output row; reconstruct only for this forward."""

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'Int8Linear':
        if (not isinstance(linear, nn.Linear) or linear.weight.dtype != torch.float32
                or linear.weight.device.type not in ('cpu', 'cuda')
                or not torch.isfinite(linear.weight).all()
                or (linear.bias is not None and (linear.bias.dtype != torch.float32
                    or linear.bias.device != linear.weight.device or not torch.isfinite(linear.bias).all()))):
            raise ValueError('Int8 conversion requires finite FP32 Linear weights/bias on CPU or CUDA')
        with torch.no_grad():
            weight = linear.weight.double()
            maximum = weight.abs().amax(dim=1, keepdim=True)
            scale = (maximum / 127).clamp_min(torch.finfo(torch.float32).tiny).float()
            scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
            qweight = (weight / scale.double()).round().clamp(-127, 127).to(torch.int8)
            if not torch.isfinite(qweight.float() * scale).all():
                raise ValueError('Reconstructed int8 weights must remain finite in FP32')
            layer = cls()
            layer.in_features, layer.out_features = linear.in_features, linear.out_features
            layer.register_buffer('qweight', qweight)
            layer.register_buffer('scale', scale)
            layer.register_buffer('bias', linear.bias.detach().clone() if linear.bias is not None else None)
        return layer.train(linear.training)

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (not isinstance(input, torch.Tensor) or input.ndim < 1
                or input.shape[-1] != self.in_features or input.dtype != torch.float32
                or input.device != self.qweight.device or self.scale.dtype != torch.float32
                or (self.bias is not None and self.bias.dtype != torch.float32)):
            raise ValueError('Int8Linear needs same-device FP32 input with matching final dimension')
        # ponytail: reconstruct a layer every forward; fused integer kernels only if measurements justify them.
        weight = self.qweight.float() * self.scale
        return F.linear(input, weight, self.bias)


def quantize_model(model: GPT2Model) -> GPT2Model:
    """Convert block projections in place, preserving the FP32 tied head."""
    if not isinstance(model, GPT2Model):
        raise ValueError('Quantization requires a custom GPT2Model')
    targets = []
    cfg = model.config
    for block in model.blocks:
        targets.extend([(block.attention, 'qkv', cfg.hidden_size, 3 * cfg.hidden_size),
                        (block.attention, 'projection', cfg.hidden_size, cfg.hidden_size),
                        (block.mlp, 'up', cfg.hidden_size, cfg.intermediate_size),
                        (block.mlp, 'down', cfg.intermediate_size, cfg.hidden_size)])
    layers = [getattr(parent, name) for parent, name, _, _ in targets]
    converted = all(isinstance(layer, Int8Linear) for layer in layers)
    kind = Int8Linear if converted else nn.Linear
    device = model.token_embedding.weight.device
    if (len(model.blocks) != cfg.num_layers or model.token_embedding.weight.dtype != torch.float32
            or model.lm_head.weight is not model.token_embedding.weight):
        raise ValueError('Quantization requires compatible FP32 embeddings and tied head')
    for layer, (_, _, inputs, outputs) in zip(layers, targets):
        if (not isinstance(layer, kind) or layer.in_features != inputs or layer.out_features != outputs):
            raise ValueError('Quantization requires complete compatible projection topology')
        weight = layer.qweight if converted else layer.weight
        if weight.shape != (outputs, inputs) or weight.device != device:
            raise ValueError('Projection weights must have matching dimensions/device')
        if converted:
            if (weight.dtype != torch.int8 or layer.scale.shape != (outputs, 1)
                    or layer.scale.dtype != torch.float32 or layer.scale.device != device
                    or not torch.isfinite(layer.scale).all() or not (layer.scale > 0).all()):
                raise ValueError('Converted projections must have compatible int8/FP32 buffers')
    if converted:
        return model
    # Prepare everything before attaching: a late invalid source cannot leave a mixed model.
    replacements = [Int8Linear.from_linear(layer) for layer in layers]
    for (parent, name, _, _), replacement in zip(targets, replacements):
        setattr(parent, name, replacement)
    return model


def model_storage_bytes(model: GPT2Model) -> dict[str, int]:
    """Actual unique storages; tied parameters/views never count twice."""
    def storage(tensor):
        data = tensor.untyped_storage()
        return (tensor.device, data.data_ptr(), data.nbytes())
    weights = {storage(parameter) for parameter in model.parameters()}
    maximum = 0
    for module in model.modules():
        if isinstance(module, Int8Linear):
            weights.update(storage(buffer) for buffer in module.buffers() if buffer is not None)
            maximum = max(maximum, module.qweight.numel() * 4)
    all_storages = weights | {storage(buffer) for buffer in model.buffers()}
    return {'weight_bytes': sum(size for _, _, size in weights),
            'model_bytes': sum(size for _, _, size in all_storages),
            'max_reconstruction_bytes': maximum}
