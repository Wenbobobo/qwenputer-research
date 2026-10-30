"""Stage-3 idea A: rebuild the 12 chip layers of the published Qwenputer checkpoint from the
upstream compiler (chipformer, rv-store branch) and compare them bit-for-bit with the shards.

Never instantiates Qwen: everything Qwen-derived that the merge needs (text_config, the base
embedding's first d0 columns for text_rows) is read from the published checkpoint itself.

    set PYTHONPATH=repro/qwenputer/experimental/riscv
    set HF_HUB_OFFLINE=1
    .venv\\Scripts\\python.exe analysis/rebuild/rebuild_chip.py [--model DIR] [--out report.json]
"""

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def _pmc():
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = wintypes.HANDLE
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    ctypes.WinDLL("psapi").GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p


def rss_gb() -> float:
    try:
        return _pmc().WorkingSetSize / 2**30
    except Exception:  # noqa: BLE001
        return float("nan")


def peak_gb() -> dict:
    try:
        p = _pmc()
        return {"peak_working_set_gb": round(p.PeakWorkingSetSize / 2**30, 2),
                "peak_commit_gb": round(p.PeakPagefileUsage / 2**30, 2)}
    except Exception:  # noqa: BLE001
        return {}


class Shards:
    def __init__(self, model: Path):
        self.model = model
        self.wm = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        self.fh = {}

    def _f(self, k):
        f = self.wm[k]
        if f not in self.fh:
            self.fh[f] = safe_open(str(self.model / f), "pt")
        return self.fh[f]

    def get(self, k: str) -> torch.Tensor:
        return self._f(k).get_tensor(k)

    def get_slice(self, k: str):
        return self._f(k).get_slice(k)


def ranges(idx: np.ndarray, limit: int = 12) -> list:
    """Compress ints into [lo, hi] runs (first limit)."""
    idx = np.unique(idx)
    out = []
    if len(idx) == 0:
        return out
    lo = prev = int(idx[0])
    for x in idx[1:]:
        x = int(x)
        if x != prev + 1:
            out.append([lo, prev])
            lo = x
        prev = x
    out.append([lo, prev])
    return out[:limit] + (["..."] if len(out) > limit else [])


def compare(name: str, mine: torch.Tensor, pub: torch.Tensor) -> dict:
    r = {"name": name, "shape": list(pub.shape), "dtype_pub": str(pub.dtype),
         "shape_match": list(mine.shape) == list(pub.shape)}
    if not r["shape_match"]:
        r.update(equal=False, rebuilt_shape=list(mine.shape))
        return r
    mine = mine.to(pub.dtype)
    if pub.dtype == torch.bfloat16:  # bitwise (also catches -0.0 vs 0.0 and NaN payloads)
        bad = mine.contiguous().view(torch.int16) != pub.contiguous().view(torch.int16)
    else:
        bad = mine != pub
    n_bad = int(bad.sum())
    r["equal"] = n_bad == 0
    r["torch_equal"] = bool(torch.equal(mine, pub))
    r["nonzero_pub"] = int((pub != 0).sum())
    import hashlib

    r["sha256_pub"] = hashlib.sha256(pub.contiguous().view(torch.uint8).numpy().tobytes()
                                     if pub.dtype == torch.bfloat16 else
                                     pub.contiguous().numpy().tobytes()).hexdigest()
    r["mismatch_count"] = n_bad
    if n_bad:
        diff = (mine.float() - pub.float())
        r["max_abs_diff"] = float(diff.abs().max())
        nz = bad.nonzero().numpy()
        if nz.shape[1] >= 1:
            r["mismatch_rows"] = ranges(nz[:, 0])
        if nz.shape[1] >= 2:
            r["mismatch_cols"] = ranges(nz[:, 1])
        ex = []
        for t in nz[:8]:
            t = tuple(int(x) for x in t)
            ex.append({"idx": list(t), "rebuilt": float(mine[t]), "published": float(pub[t])})
        r["examples"] = ex
    else:
        r["max_abs_diff"] = 0.0
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".data/Qwen3.5-4B-Qwenputer-Filesystem")
    ap.add_argument("--out", default="analysis/rebuild/report.json")
    ap.add_argument("--skip-emb", action="store_true")
    ap.add_argument("--skip-text-rows", action="store_true")
    ap.add_argument("--store-seed", type=int, help="negative control: override store seed")
    args = ap.parse_args()
    model = Path(args.model)

    from chipformer.q35.emit import Layout, allocate, emit_layers
    from chipformer.q35.ir import Target
    from chipformer.q35.merge import TAU_GAIN
    from chipformer.q35.tiny import code_gain
    from chipformer.rv import bridge as BR
    from chipformer.rv import core
    from chipformer.rv import spec as S
    from chipformer.rv.merge import ID_GAIN, chip_token_names, text_rows
    from chipformer.rv.store import Config
    from chipformer.rv.tiny import embedding_matrix

    meta = json.loads((model / "chipformer.json").read_text())
    tc = json.loads((model / "config.json").read_text())
    report = {"model": str(model), "torch": torch.__version__, "python": sys.version,
              "numpy": np.__version__, "checks": {}, "layers": [], "embedding": {},
              "final_norm": {}, "notes": []}
    v0, d0, hidden, m = (meta["chip_token_offset"], meta["chip_dims_offset"], meta["hidden"],
                         meta["ballast_dims"])

    # --- the bridge's view of Qwen's vocabulary: merge.py used the ORIGINAL tokenizer (src);
    # recover it from the extended one by cutting at the first <|reserved_0|> pad token.
    T = BR.Text(model / "tokenizer.json")
    n_ext = T.n
    v = T.tok.get_vocab(with_added_tokens=True)
    n_orig = v.get("<|reserved_0|>", v0)
    T.n = n_orig
    T.bytes = T.bytes[:n_orig]
    names = chip_token_names()
    report["checks"]["tokenizer"] = {
        "n_extended": n_ext, "n_original_recovered": n_orig, "chip_token_offset": v0,
        "tool_call": [T.TCO, T.TCC], "tool_call_match": [T.TCO, T.TCC] == meta["tool_call"],
        "printable_match": T.printable == meta["printable"],
        "chip_token_names_match": all(v.get(s) == v0 + i for i, s in enumerate(names))}
    log("tokenizer", report["checks"]["tokenizer"])

    # --- circuit
    scfg = dict(meta["store"])
    if args.store_seed is not None:
        scfg["seed"] = args.store_seed
        report["notes"].append(f"NEGATIVE CONTROL: store seed overridden to {args.store_seed}")
    store = Config(**scfg)
    target = Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"],
                    tc["intermediate_size"])
    c = core.build(target, T, store)
    keep = set(c.keep)
    alloc = allocate(c, None, keep, tc["intermediate_size"])
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots)
    lay.scale  # noqa: B018
    stats = c.stats()
    code = [next(iter(e.terms)) for e in c.code]
    tau = next(iter(c.tau.terms))
    mu = next(iter(c.mu.terms))
    id_dims = [lay.dim(next(iter(e.terms))) for e in c.store.yc]
    report["checks"]["circuit"] = {
        "stats": stats, "stats_match": stats == meta["circuit"],
        "slots": alloc.n_slots, "slots_match": alloc.n_slots == meta["slots"],
        "chip_layers": c.n_layers(), "chip_layers_match": c.n_layers() == meta["chip_layers"],
        "code_dims_match": [lay.dim(d) for d in code] == meta["code_dims"],
        "tau_dim_match": lay.dim(tau) == meta["tau_dim"],
        "mu_dim_match": lay.dim(mu) == meta["mu_dim"],
        "id_code_dims_match": id_dims == meta["id_code_dims"],
        "store_cfg": dataclasses.asdict(store),
    }
    log("circuit", report["checks"]["circuit"], f"rss {rss_gb():.1f}G")

    # comparator self-test: a single flipped bf16 LSB must be detected
    a = torch.zeros(4, 4, dtype=torch.bfloat16)
    b = a.clone()
    b.view(torch.int16)[1, 2] = 1  # smallest subnormal
    z = a.clone()
    z.view(torch.int16)[0, 0] = -32768  # -0.0
    report["checks"]["comparator_selftest"] = {
        "lsb_flip_detected": compare("t", a, b)["mismatch_count"] == 1,
        "neg_zero_detected": compare("t", a, z)["mismatch_count"] == 1}

    # --- chip layers (emit_layers returns all 12 in fp32; ~1.3 GB/layer)
    sh = Shards(model)
    n_chip = c.n_layers()
    layers = emit_layers(c, lay, tc, 0, "model.", alloc, keep)
    log(f"emitted {len(layers)} tensors, rss {rss_gb():.1f}G")
    for L in range(n_chip):
        p = f"model.layers.{L}."
        rows = []
        for k in sorted(kk for kk in layers if kk.startswith(p)):
            mine = layers.pop(k).to(torch.bfloat16)
            pub = sh.get(k)
            rows.append(compare(k, mine, pub))
            del mine, pub
        pub_keys = sorted(kk for kk in sh.wm if kk.startswith(p))
        extra = sorted(set(pub_keys) - {r["name"] for r in rows})
        ok = all(r["equal"] for r in rows) and not extra
        report["layers"].append({"layer": L, "bit_exact": ok, "tensors": rows,
                                 "published_only_keys": extra})
        gc.collect()
        log(f"layer {L}: {'BIT-EXACT' if ok else 'MISMATCH'} "
            f"({sum(r['equal'] for r in rows)}/{len(rows)} equal), rss {rss_gb():.1f}G")
    assert not layers
    del layers
    gc.collect()

    # --- final norm (chip dims only; Qwen part is Qwen's norm rescaled)
    pub_norm = sh.get("model.norm.weight")
    norm = pub_norm.float().clone()
    norm[d0:] = -1.0
    for d in code:
        norm[lay.dim(d)] = code_gain(lay)
    for d in id_dims:
        norm[d] = ID_GAIN * code_gain(lay)
    norm[lay.dim(tau)] = TAU_GAIN
    report["final_norm"] = compare("model.norm.weight[2560:]", norm[d0:].to(torch.bfloat16),
                                   pub_norm[d0:])
    log("final norm", report["final_norm"]["equal"])

    if not args.skip_emb:
        sl = sh.get_slice("model.embed_tokens.weight")
        vocab = sl.get_shape()[0]
        chip = embedding_matrix(c, lay, S.VOCAB)
        chip = torch.from_numpy(chip) if isinstance(chip, np.ndarray) else chip
        chip = chip[:S.VOCAB].to(torch.bfloat16)
        pub = sl[v0:v0 + S.VOCAB]
        report["embedding"]["chip_rows"] = compare(
            f"embed_tokens[{v0}:{v0 + S.VOCAB}] (embedding_matrix, as merge.py)", chip, pub)
        del chip
        gc.collect()
        # the exact path (embedding_rows) on a sample, vs the published rows
        from chipformer.q35.emit import embedding_rows

        rng = np.random.default_rng(0)
        samp = sorted(set(rng.integers(0, S.VOCAB, 2000).tolist()) | {0, S.VOCAB - 1})
        er = embedding_rows(c, lay, samp, hidden).to(torch.bfloat16)
        report["embedding"]["embedding_rows_sample_vs_published"] = compare(
            f"embedding_rows({len(samp)} sampled chip ids)", er, pub[samp])
        del pub, er
        gc.collect()
        log("chip rows", report["embedding"]["chip_rows"]["equal"], f"rss {rss_gb():.1f}G")
        tail = sl[v0 + S.VOCAB:vocab]
        report["embedding"]["pad_rows_after_chip"] = {
            "rows": [v0 + S.VOCAB, vocab], "all_zero": bool((tail == 0).all())}
        del tail
        if not args.skip_text_rows:
            base = sl[0:v0]
            full = text_rows(c, lay, T, base, v0)
            report["embedding"]["text_rows_chip_dims"] = compare(
                f"embed_tokens[0:{v0}, {d0}:]", full[:, d0:], base[:, d0:])
            report["embedding"]["text_rows_qwen_dims_note"] = (
                "Qwen dims [0:2560] are copied from the published tensor itself (no original "
                "Qwen loaded), so only the chip dims are an independent check")
            del base, full
            gc.collect()
            log("text rows", report["embedding"]["text_rows_chip_dims"]["equal"],
                f"rss {rss_gb():.1f}G")

    report["verdict"] = {
        "all_chip_layers_bit_exact": all(x["bit_exact"] for x in report["layers"]),
        "final_norm_chip_dims_bit_exact": report["final_norm"]["equal"],
        "chip_embedding_rows_bit_exact": report["embedding"].get("chip_rows", {}).get("equal"),
        "text_rows_chip_dims_bit_exact":
            report["embedding"].get("text_rows_chip_dims", {}).get("equal"),
    }
    report["elapsed_s"] = round(time.time() - T0, 1)
    report["memory"] = peak_gb()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    log("verdict", report["verdict"])


if __name__ == "__main__":
    main()
