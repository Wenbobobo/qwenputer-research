"""CPU-only compact-head checks, including real checkpoint head quantization."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from gpu.trim.int8_head import Int8Head


def synthetic():
    torch.manual_seed(17)
    v0, rows, d, q = 31, 47, 16, 8
    w = torch.randn(rows, d, dtype=torch.bfloat16)
    w[v0:, :q] = 0
    nw = torch.full((d,), -1., dtype=torch.bfloat16)
    nw[:q] = 0
    nw[10] = 7
    nw[15] = 3
    x = torch.randn(2, 3, d, dtype=torch.bfloat16)
    x[..., q:] = 0
    x[..., 10] = 5
    x[..., 15] = -3
    h = Int8Head(w, nw, v0=v0, qwen_dims=q, chunk_rows=7)
    y = h(x)
    ref = F.linear(x, w)
    assert y.shape == ref.shape
    assert torch.equal(y[..., v0:], ref[..., v0:])
    assert torch.allclose(y[..., :v0].float(), ref[..., :v0].float(), atol=.25, rtol=.02)
    # Noncontiguous input and a zero quantized row; partial final chunk.
    w[0, :q] = 0
    h = Int8Head(w, nw, v0=v0, qwen_dims=q, text_only=True, chunk_rows=7)
    assert torch.isfinite(h.scales).all()
    assert torch.isneginf(h(x.transpose(0, 1))[..., v0:]).all()
    try:
        Int8Head(w, nw, v0=v0, qwen_dims=q, chunk_rows=0)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid chunk size accepted')
    return {'passed': True, 'shape': list(y.shape), 'chip_rows_exact': True}


def real_head(root):
    idx = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    def load(key):
        with safe_open(str(root / idx[key]), framework='pt', device='cpu') as f:
            return f.get_tensor(key)
    emb_key = next(k for k in idx if k.endswith('embed_tokens.weight'))
    norm_key = next(k for k in idx if k.endswith('model.norm.weight'))
    cfg = json.loads((root / 'config.json').read_text())
    assert cfg.get('tie_word_embeddings', cfg.get('text_config', {}).get('tie_word_embeddings', False))
    assert not any(k.endswith('lm_head.weight') for k in idx), 'untied checkpoint requires actual head'
    w, nw = load(emb_key), load(norm_key)
    t = time.perf_counter()
    h = Int8Head(w, nw)
    # Compare Qwen-only and circuit-only activation probes. These are head
    # structure checks, not transcript correctness or a language quality test.
    torch.manual_seed(23)
    x = torch.zeros(1, 1, w.shape[1], dtype=w.dtype)
    x[..., :2560] = torch.randn(2560).to(w.dtype)
    y, ref = h(x), F.linear(x, w)
    diff = (y.float() - ref.float()).abs()
    r = h.storage_report()
    r.update(seconds=time.perf_counter()-t, text_probe_max_abs_diff=float(diff.max()),
             text_probe_argmax=[int(ref.argmax()), int(y.argmax())])
    x.zero_()
    x[..., h.chip_cols] = torch.randn(h.chip_cols.numel()).to(w.dtype) * 10
    y, ref = h(x), F.linear(x, w)
    r.update(chip_probe_argmax=[int(ref.argmax()), int(y.argmax())],
             chip_probe_max_abs_diff=float((y.float()-ref.float()).abs().max()),
             chip_probe_rows_exact=bool(torch.equal(y[..., 248320:], ref[..., 248320:])))
    assert r['chip_probe_rows_exact'], r
    assert r['chip_probe_max_abs_diff'] == 0, r
    assert r['chip_probe_argmax'][0] == r['chip_probe_argmax'][1], r
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--output', default='gpu/out/int8_head_cpu.json')
    a = ap.parse_args()
    torch.set_num_threads(8)
    r = {'synthetic': synthetic()}
    if a.real:
        r['real_head'] = real_head(Path('.data/Qwen3.5-4B-Qwenputer-Filesystem'))
    Path(a.output).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2), flush=True)

if __name__ == '__main__':
    main()
