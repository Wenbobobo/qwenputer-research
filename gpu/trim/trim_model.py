"""Lossless Qwen-layer trimming for the Qwenputer merged checkpoint.

The merged model keeps a 4608-wide residual, but Qwen layers read only columns
0:2560 and mu_dim (2732), and write only rows 0:2560. This module keeps the HF
layer implementations (including GDN/full-attention caches) and replaces only
linear maps with packed projections.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

QWEN_START = 12
HIDDEN = 4608
QWEN_DIMS = 2560
MU_DIM = 2732


class SlimLinear(nn.Module):
    """Linear projection with selected input columns and output rows."""
    def __init__(self, weight: torch.Tensor, *, select_input: bool, trim_output: bool):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.select_input = select_input
        self.trim_output = trim_output
        if select_input:
            weight = torch.cat((weight[:, :QWEN_DIMS], weight[:, MU_DIM:MU_DIM + 1]), dim=1)
        if trim_output:
            weight = weight[:QWEN_DIMS]
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.select_input:
            x = torch.cat((x[..., :QWEN_DIMS], x[..., MU_DIM:MU_DIM + 1]), dim=-1)
        y = F.linear(x, self.weight)
        if self.trim_output:
            out = y.new_zeros(*y.shape[:-1], self.out_features)
            out[..., :QWEN_DIMS] = y
            return out
        return y


def _replace_linear(parent: nn.Module) -> tuple[int, int]:
    before = after = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.Linear) and child.bias is None:
            select_input = child.in_features == HIDDEN
            trim_output = child.out_features == HIDDEN
            if select_input or trim_output:
                slim = SlimLinear(child.weight.detach(), select_input=select_input,
                                  trim_output=trim_output)
                before += child.weight.numel()
                after += slim.weight.numel()
                setattr(parent, name, slim)
                continue
        b, a = _replace_linear(child)
        before += b
        after += a
    return before, after


def trim_model(model: nn.Module, *, qwen_start: int = QWEN_START) -> dict[str, int]:
    """Trim layers qwen_start..end in-place; leave chip layers untouched."""
    inner = getattr(model, "model", model)
    lm = getattr(inner, "language_model", inner)
    before = after = count = 0
    for idx, layer in enumerate(lm.layers):
        if idx < qwen_start:
            continue
        b, a = _replace_linear(layer)
        before += b
        after += a
        count += sum(isinstance(m, SlimLinear) for m in layer.modules())
    return {"qwen_dense_numel": before, "qwen_trimmed_numel": after,
            "saved_numel": before - after, "slim_linear_count": count}


def load_trimmed_state_dict(checkpoint: str | Path, *, qwen_start: int = QWEN_START,
                            index_name: str = "model.safetensors.index.json") -> dict[str, torch.Tensor]:
    """Read each safetensors entry lazily and trim eligible Qwen matrices."""
    root = Path(checkpoint)
    index = json.loads((root / index_name).read_text())
    result: dict[str, torch.Tensor] = {}
    for name, shard in index["weight_map"].items():
        parts = name.split(".")
        layer_idx = int(parts[2]) if len(parts) > 2 and parts[:2] == ["model", "layers"] else -1
        with safe_open(str(root / shard), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)
        if layer_idx >= qwen_start and tensor.ndim == 2:
            if tensor.shape[1] == HIDDEN:
                tensor = torch.cat((tensor[:, :QWEN_DIMS], tensor[:, MU_DIM:MU_DIM + 1]), dim=1)
            if tensor.shape[0] == HIDDEN:
                tensor = tensor[:QWEN_DIMS]
        result[name] = tensor
    return result


def build_trimmed_runtime(model: nn.Module, *, qwen_start: int = QWEN_START,
                          compact_chip: bool = False) -> dict[str, int]:
    """Trim a loaded HF model; optionally compact its chip layers."""
    stats = trim_model(model, qwen_start=qwen_start)
    if compact_chip:
        from chipformer.q35.fast import compact
        compact(model, qwen=False, n_chip=qwen_start)
    return stats


def footprint(model: nn.Module) -> dict[str, int]:
    """Return actual parameter and slim-projection byte counts."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    slim = sum(m.weight.numel() * m.weight.element_size()
               for m in model.modules() if isinstance(m, SlimLinear))
    return {"parameter_bytes": total, "slim_qwen_bytes": slim}


__all__ = ["SlimLinear", "trim_model", "build_trimmed_runtime",
           "load_trimmed_state_dict", "footprint"]
