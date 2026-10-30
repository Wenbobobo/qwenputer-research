"""Opt-in fp32 execution for the compacted chip layers (0..11).

The surrounding Qwen stack remains bf16. Chip layers carry fp32 parameters and
pass fp32 residuals between chip layers; only the final chip output is cast back
to bf16 at the residual boundary consumed by Qwen layers.
"""
from __future__ import annotations

import torch
from torch import nn


class _Fp32ChipLayer(nn.Module):
    def __init__(self, inner: nn.Module, *, final: bool):
        super().__init__()
        self.inner = inner
        self.final = final
        # compact() has already replaced dense linears; convert all owned state.
        self.inner.float()

    def __getattr__(self, name):
        # Preserve decoder metadata (linear_attn, layer_idx, etc.) for callers.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('inner'), name)

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = hidden_states.float()
        # Chip layers do not use RoPE, but preserve the generic HF call contract.
        if kwargs.get("position_embeddings") is not None:
            pe = kwargs["position_embeddings"]
            if isinstance(pe, tuple):
                kwargs["position_embeddings"] = tuple(x.float() for x in pe)
            elif torch.is_tensor(pe):
                kwargs["position_embeddings"] = pe.float()
        out = self.inner(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            hs = out[0].to(torch.bfloat16) if self.final else out[0]
            return (hs, *out[1:])
        return out.to(torch.bfloat16) if self.final else out


def install(model: nn.Module, n_chip: int = 12) -> dict:
    """Convert and wrap compacted chip layers in-place; return audit stats."""
    inner = getattr(model, "model", model)
    lm = getattr(inner, "language_model", inner)
    wrapped = 0
    fp32_bytes = 0
    for i in range(min(n_chip, len(lm.layers))):
        layer = lm.layers[i]
        if isinstance(layer, _Fp32ChipLayer):
            continue
        layer.float()
        lm.layers[i] = _Fp32ChipLayer(layer, final=(i == n_chip - 1))
        wrapped += 1
        fp32_bytes += sum(p.numel() * 4 for p in lm.layers[i].parameters())
        fp32_bytes += sum(b.numel() * 4 for b in lm.layers[i].buffers())
    model._trim_fp32_chip = True
    return {"enabled": True, "layers": wrapped, "fp32_bytes": fp32_bytes,
            "output_dtype": "bf16", "input_dtype": "fp32"}


__all__ = ["install"]
