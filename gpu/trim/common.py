"""Shared helpers for the trimmed-Qwen RvChat runtime (gpu/trim). No repro/.data edits."""
import sys
from pathlib import Path
import os
# flash-linear-attention (fla 0.6.0) appeared in the venv at 20:10; transformers binds it at import
# time for every GDN layer, including chip layers 0-11 whose arithmetic must stay exact. Block it
# (reference torch path, as in the Stage-5 baseline) unless TRIM_ALLOW_FLA=1.
if os.environ.get('TRIM_ALLOW_FLA') != '1' and os.environ.get('TRIM_SELECTIVE_FLA') != '1':
    sys.modules.setdefault('fla', None)
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'repro/qwenputer/experimental/riscv'), str(ROOT/'repro/qwenputer'), str(ROOT)]
from gpu.trim.trim_model import trim_model, SlimLinear, QWEN_START, HIDDEN, QWEN_DIMS, MU_DIM
CKPT = ROOT/'.data/Qwen3.5-4B-Qwenputer-Filesystem'


def check_dropped_zero(model, qwen_start=QWEN_START):
    """Prove losslessness on the real checkpoint: every weight entry trim_model drops is 0."""
    lm = model.model
    lm = getattr(lm, 'language_model', lm)
    bad, checked = [], 0
    for i in range(qwen_start, len(lm.layers)):
        for name, m in lm.layers[i].named_modules():
            if not isinstance(m, nn.Linear) or m.bias is not None:
                continue
            w = m.weight
            if w.shape[1] == HIDDEN:
                checked += 1
                nz = int((w[:, QWEN_DIMS:MU_DIM] != 0).sum() + (w[:, MU_DIM + 1:] != 0).sum())
                if nz: bad.append((f'layers.{i}.{name}', 'cols', nz))
            if w.shape[0] == HIDDEN:
                checked += 1
                nz = int((w[QWEN_DIMS:] != 0).sum())
                if nz: bad.append((f'layers.{i}.{name}', 'rows', nz))
    return {'checked_matrices': checked, 'nonzero_dropped': bad}


class SlimHead(nn.Module):
    """Optional lm_head variant: logits of the text vocabulary only (rows < v0), over the
    nonzero column hull; chip rows are filled with -inf (the circuit owns chip tokens).
    Breaks only the head_mismatch diagnostic (Qwen never predicts a chip id)."""
    def __init__(self, w, v0):
        super().__init__()
        nzc = (w[:v0] != 0).any(0).nonzero().flatten()
        self.c0, self.c1 = int(nzc[0]), int(nzc[-1]) + 1
        self.v0, self.out_features = v0, w.shape[0]
        self.weight = nn.Parameter(w[:v0, self.c0:self.c1].contiguous(), requires_grad=False)
    def forward(self, x):
        y = torch.nn.functional.linear(x[..., self.c0:self.c1], self.weight)
        out = y.new_full((*y.shape[:-1], self.out_features), float('-inf'))
        out[..., :self.v0] = y
        return out
