"""Stage-3 A2 (optional): re-fit the anti-sink keys on CPU with upstream q35/mute.fit_anti_sink and
compare them with the k_proj[:, mu] columns of the published checkpoint (stored in base_report.json).

The fit runs Qwen3.5-4B (bf16) over report.PROMPTS and averages q_norm outputs, so it depends on the
device/kernels/library versions; bit-exactness is NOT expected. We report bitwise equality, the
bf16 element-diff distribution and the cosine per kv head.

    set PYTHONPATH=repro\\qwenputer\\experimental\\riscv
    repro\\qwenputer\\experimental\\riscv\\.venv\\Scripts\\python.exe analysis/rebuild/audit_base_sink.py
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

T0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".data/Qwen3.5-4B")
    ap.add_argument("--report", default="analysis/rebuild/base_report.json")
    args = ap.parse_args()
    from chipformer.q35.mute import fit_anti_sink

    rep = json.loads(Path(args.report).read_text())
    hidden, n_chip = rep["hidden"], rep["n_chip"]
    torch.set_num_threads(os.cpu_count() or 8)
    sinks = fit_anti_sink(args.base, "cpu")
    nb = hidden ** 0.5
    out = {"device": "cpu", "torch": torch.__version__, "layers": {}}
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    all_bit, tot, eq = True, 0, 0
    for qi, keys in sorted(sinks.items()):
        layer = qi + n_chip
        col = (keys.flatten() / nb).to(torch.bfloat16)  # exactly mute()'s store expression
        pub_bits = torch.tensor(rep["anti_sink_columns_bf16_hex"][str(layer)], dtype=torch.int32)
        pub = pub_bits.to(torch.int16).view(torch.bfloat16)  # wraps 0x8000.. to negative int16
        same = (col.view(torch.int16).to(torch.int32) & 0xFFFF) == pub_bits
        nkv = keys.shape[0]
        cos = torch.nn.functional.cosine_similarity(col.float().view(nkv, -1),
                                                    pub.float().view(nkv, -1), dim=1)
        d = (col.float() - pub.float()).abs()
        rel = d.max() / pub.float().abs().max()
        out["layers"][str(layer)] = {"bit_equal_elems": int(same.sum()), "elems": col.numel(),
                                     "cos_per_kv_head": [round(float(c), 6) for c in cos],
                                     "max_abs_diff": float(d.max()), "max_rel_to_colmax": float(rel)}
        tot += col.numel()
        eq += int(same.sum())
        all_bit &= bool(same.all())
        print(layer, out["layers"][str(layer)], flush=True)
    out["all_bit_exact"] = all_bit
    out["bit_equal_fraction"] = eq / tot
    out["min_cos"] = min(min(v["cos_per_kv_head"]) for v in out["layers"].values())
    out["seconds"] = round(time.time() - T0, 1)
    rep["anti_sink_refit_cpu"] = out
    Path(args.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "layers"}, indent=1))


if __name__ == "__main__":
    main()
