"""Opt-in weight-only INT8 output projection; embedding and chip features stay BF16.

Only text-row Qwen columns are quantized. Final-norm live chip columns are kept
exact, including chip rows, so circuit logits retain their large exact margins.
This adds a compact head to a tied BF16 embedding; it does NOT save resident VRAM.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Int8Head(nn.Module):
    def __init__(self, weight, norm_weight, *, v0=248320, qwen_dims=2560,
                 text_only=False, chunk_rows=16384):
        super().__init__()
        if chunk_rows <= 0:
            raise ValueError('chunk_rows must be positive')
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.v0 = v0
        self.qwen_dims = qwen_dims
        self.text_only = text_only
        self.chunk_rows = chunk_rows
        if norm_weight.numel() != self.in_features:
            raise ValueError('final norm and head dimensions differ')
        # Qwen3.5 RMSNorm uses (1 + weight), not weight itself.
        chip_cols = (norm_weight[qwen_dims:].float() != -1).nonzero().flatten() + qwen_dims
        self.register_buffer('chip_cols', chip_cols)
        rows = v0 if text_only else self.out_features
        self.register_buffer('chip_weight', weight[:rows].index_select(1, chip_cols).contiguous())
        self.register_buffer('qweight', torch.empty((v0, qwen_dims), dtype=torch.int8, device=weight.device))
        self.register_buffer('scales', torch.empty((v0, 1), dtype=torch.float32, device=weight.device))
        # Construction is chunked too: never cast the entire 4.47 GB embedding.
        with torch.no_grad():
            for lo in range(0, v0, chunk_rows):
                hi = min(lo + chunk_rows, v0)
                w = weight[lo:hi, :qwen_dims].float()
                scale = w.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
                self.qweight[lo:hi] = (w / scale).round().clamp(-127, 127).to(torch.int8)
                self.scales[lo:hi] = scale
            # Chip rows must have no Qwen contribution for this factorization.
            for lo in range(v0, self.out_features, chunk_rows):
                if torch.count_nonzero(weight[lo:lo + chunk_rows, :qwen_dims]).item():
                    raise ValueError('chip rows contain Qwen weights')

    def storage_report(self):
        return dict(quantized_rows=self.v0, quantized_cols=self.qwen_dims,
                    exact_chip_cols=self.chip_cols.numel(), text_only=self.text_only,
                    chunk_rows=self.chunk_rows,
                    buffer_bytes=sum(b.numel() * b.element_size() for b in self.buffers()),
                    dequant_chunk_bytes=min(self.v0, self.chunk_rows) * self.qwen_dims * 2,
                    embedding='unchanged BF16 tied original; separate compact head adds storage')

    def forward(self, x):
        # Compute each row in one GEMM (including exact chip columns) to avoid
        # separate BF16 rounding of the text and circuit contributions.
        if x.dtype != self.chip_weight.dtype:
            raise TypeError('Int8Head activation dtype must match stored exact chip weights')
        xc = x.index_select(-1, self.chip_cols)
        xk = torch.cat((x[..., :self.qwen_dims], xc), dim=-1)
        out = x.new_full((*x.shape[:-1], self.out_features), float('-inf'))
        for lo in range(0, self.v0, self.chunk_rows):
            hi = min(lo + self.chunk_rows, self.v0)
            w = self.qweight[lo:hi].to(x.dtype)
            w.mul_(self.scales[lo:hi].to(x.dtype))
            packed = torch.cat((w, self.chip_weight[lo:hi]), dim=-1)
            out[..., lo:hi] = F.linear(xk, packed)
            del w, packed
        if not self.text_only:
            out[..., self.v0:] = F.linear(xc, self.chip_weight[self.v0:])
        return out


def install_int8_head(model, *, text_only=False, chunk_rows=16384):
    inner = model.model
    lm = getattr(inner, 'language_model', inner)
    head = Int8Head(model.lm_head.weight.detach(), lm.norm.weight.detach(),
                    text_only=text_only, chunk_rows=chunk_rows)
    model.lm_head = head
    return head.storage_report()
