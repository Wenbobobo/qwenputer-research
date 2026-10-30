"""Stage-3 A2: audit the Qwen side of the published Qwenputer checkpoint against the base model.

For every published tensor that comes from Qwen3.5-4B (layers 12..43, the final norm's first d0
dims, the embedding's [0:v0, 0:d0] block) compute the expected tensor from the base checkpoint with
upstream chipformer's merge rules (rv/merge.py + q35/merge.py IN_COLS / OUT_ROWS / WIDE_NORMS) and
compare bf16 bit patterns. Then localize every mismatch and test it against q35/mute.py:

  linear-attention layers: in_proj_a/b/z[:, mu] = bf16(-1024 / sqrt(hidden))
  full-attention layers:   q_proj[gate rows of every head, mu] = the same constant
                           k_proj[:, mu] = anti_sink / sqrt(hidden)   (fitted, device-dependent)

The rules are re-implemented here (not imported) so the audit is independent of the code under
audit; they are copied verbatim in spirit from the files named above.

    repro\\qwenputer\\experimental\\riscv\\.venv\\Scripts\\python.exe analysis/rebuild/audit_base.py
        [--base .data/Qwen3.5-4B] [--model .data/Qwen3.5-4B-Qwenputer-Filesystem]
        [--out analysis/rebuild/base_report.json]
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

T0 = time.time()

# --- the merge rules (q35/merge.py, rv/merge.py, q35/mute.py) ---------------------------------
IN_COLS = re.compile(
    r"(in_proj_(qkv|z|a|b)|q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$|embed_tokens\.weight$")
OUT_ROWS = re.compile(r"(out_proj|o_proj|down_proj)\.weight$|visual\.merger\.linear_fc2\.(weight|bias)$")
WIDE_NORMS = re.compile(r"(input_layernorm|post_attention_layernorm)\.weight$|language_model\.norm\.weight$")
SRC_TEXT, DST = "model.language_model.", "model."
MUTE = -1024.0
ROTARY = 64
SINK_NORM = 16.0


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def rss_gb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        p = PMC()
        p.cb = ctypes.sizeof(PMC)
        ctypes.WinDLL("psapi").GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return {"ws_gb": round(p.WorkingSetSize / 2**30, 2),
                "peak_ws_gb": round(p.PeakWorkingSetSize / 2**30, 2),
                "peak_commit_gb": round(p.PeakPagefileUsage / 2**30, 2)}
    except Exception:  # noqa: BLE001
        return {}


class Shards:
    def __init__(self, d: Path):
        self.d = d
        idx = d / "model.safetensors.index.json"
        self.wm = json.loads(idx.read_text())["weight_map"]
        self.fh = {}

    def _f(self, k):
        f = self.wm[k]
        if f not in self.fh:
            self.fh[f] = safe_open(str(self.d / f), "pt")
        return self.fh[f]

    def get(self, k):
        return self._f(k).get_tensor(k)

    def slice(self, k):
        return self._f(k).get_slice(k)

    def keys(self):
        return list(self.wm)


def bits(t):
    iv = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}
    return t.contiguous().view(iv[t.dtype])


def diff_stats(exp, pub):
    """bitwise mismatch mask + summary."""
    assert exp.dtype == pub.dtype, (exp.dtype, pub.dtype)
    m = bits(exp) != bits(pub)
    n = int(m.sum())
    out = {"numel": exp.numel(), "mismatch": n, "equal": n == 0}
    if n:
        d = (exp.float() - pub.float()).abs()
        out["max_abs_diff"] = float(d[m].max())
    else:
        out["max_abs_diff"] = 0.0
    return m, out


def localize(m):
    """which columns/rows the mismatches sit in (2-D: dim1 = hidden cols for IN_COLS)."""
    if m.dim() == 1:
        idx = m.nonzero().flatten().tolist()
        return {"positions": idx[:32], "n_positions": len(idx)}
    cols = m.any(0).nonzero().flatten().tolist()
    rows = m.any(1).nonzero().flatten().tolist()
    return {"cols": cols[:32], "n_cols": len(cols), "rows_first": rows[:8], "rows_last": rows[-4:],
            "n_rows": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".data/Qwen3.5-4B")
    ap.add_argument("--model", default=".data/Qwen3.5-4B-Qwenputer-Filesystem")
    ap.add_argument("--out", default="analysis/rebuild/base_report.json")
    ap.add_argument("--chunk", type=int, default=16384, help="embedding rows per comparison chunk")
    args = ap.parse_args()
    base_dir, pub_dir = Path(args.base), Path(args.model)

    bcfg = json.loads((base_dir / "config.json").read_text())
    tc0 = bcfg["text_config"]
    pcfg = json.loads((pub_dir / "config.json").read_text())
    meta = json.loads((pub_dir / "chipformer.json").read_text())
    d0, v0, n0 = tc0["hidden_size"], tc0["vocab_size"], tc0["num_hidden_layers"]
    hidden, n_chip, mu = meta["hidden"], meta["chip_layers"], meta["mu_dim"]
    hd = tc0["head_dim"]
    assert pcfg["hidden_size"] == hidden and meta["chip_dims_offset"] == d0
    assert pcfg["num_hidden_layers"] == n_chip + n0
    assert pcfg["layer_types"][n_chip:] == tc0["layer_types"], "layer_types tail != base"
    scale = (d0 / hidden) ** 0.5
    normed_ballast = hidden ** 0.5
    w_mute = torch.tensor(MUTE / normed_ballast, dtype=torch.float64).to(torch.bfloat16)
    log(f"d0={d0} v0={v0} hidden={hidden} n_chip={n_chip} mu={mu} scale={scale!r} "
        f"mute const {MUTE / normed_ballast!r} -> bf16 {float(w_mute)!r}")

    B, P = Shards(base_dir), Shards(pub_dir)
    pub_keys = P.keys()
    base_keys = B.keys()
    report = {"base": str(base_dir), "model": str(pub_dir), "d0": d0, "v0": v0, "hidden": hidden,
              "n_chip": n_chip, "mu_dim": mu, "norm_scale": scale,
              "mute_const": MUTE / normed_ballast, "mute_const_bf16": float(w_mute),
              "n_published_tensors": len(pub_keys), "n_base_tensors": len(base_keys)}

    # ---- key mapping ---------------------------------------------------------------------
    def to_pub(k):
        k2 = DST + k[len(SRC_TEXT):]
        return re.sub(r"^(model\.layers\.)(\d+)\.", lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k2)

    text_keys = [k for k in base_keys if k.startswith(SRC_TEXT)]
    skipped_base = sorted({re.sub(r"\.\d+\.", ".N.", k).split(".")[0] + "." + k.split(".")[1]
                           for k in base_keys if not k.startswith(SRC_TEXT)})
    mapping = {to_pub(k): k for k in text_keys}
    emb_pub, norm_pub = DST + "embed_tokens.weight", DST + "norm.weight"
    chip_re = re.compile(r"^model\.layers\.(\d+)\.")
    chip_keys = [k for k in pub_keys if chip_re.match(k) and int(chip_re.match(k)[1]) < n_chip]
    qwen_layer_keys = [k for k in pub_keys if chip_re.match(k) and int(chip_re.match(k)[1]) >= n_chip]
    other = sorted(set(pub_keys) - set(chip_keys) - set(qwen_layer_keys))
    missing = sorted(set(mapping) - set(pub_keys))
    extra = sorted(set(qwen_layer_keys) - set(mapping))
    report["keys"] = {"chip_layer_tensors": len(chip_keys), "qwen_layer_tensors": len(qwen_layer_keys),
                      "other_tensors": other, "base_text_tensors": len(text_keys),
                      "base_non_text_prefixes_skipped": skipped_base,
                      "mapped_missing_in_published": missing, "published_qwen_without_base": extra}
    log(f"published {len(pub_keys)} tensors: chip {len(chip_keys)}, qwen-layers "
        f"{len(qwen_layer_keys)}, other {other}; missing {missing} extra {extra}")

    # ---- per-tensor audit of the Qwen layers ----------------------------------------------
    rows = []
    sinks = {}
    full_layers = [n_chip + i for i in range(n0) if tc0["layer_types"][i] == "full_attention"]
    for pk in sorted(qwen_layer_keys, key=lambda k: (int(chip_re.match(k)[1]), k)):
        bk = mapping[pk]
        layer = int(chip_re.match(pk)[1])
        w = B.get(bk)
        rule = "copy"
        if IN_COLS.search(bk):
            exp = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
            rule = "in_cols_pad"
        elif OUT_ROWS.search(bk):
            exp = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
            rule = "out_rows_pad"
        elif WIDE_NORMS.search(bk):
            g = (1.0 + w.float()) * scale - 1.0
            exp = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])
            rule = "wide_norm_rescale_pad"
        else:
            exp = w
        pub = P.get(pk)
        r = {"key": pk, "base_key": bk, "layer": layer, "rule": rule,
             "type": pcfg["layer_types"][layer], "shape": list(pub.shape),
             "shape_ok": list(exp.shape) == list(pub.shape), "dtype": str(pub.dtype)}
        if not r["shape_ok"]:
            r["error"] = f"shape {list(exp.shape)} vs {list(pub.shape)}"
            rows.append(r)
            continue
        m, st = diff_stats(exp, pub)
        r.update(st)
        r["class"] = "deterministic" if st["equal"] else "UNEXPLAINED"
        if not st["equal"]:
            r["where"] = localize(m)
            # outside the mu column?
            off_mu = m.clone()
            if m.dim() == 2:
                off_mu[:, mu] = False
            r["mismatch_outside_mu"] = int(off_mu.sum())
            name = pk.split(".")[-2]
            col_pub = pub[:, mu] if m.dim() == 2 else None
            col_base = exp[:, mu] if m.dim() == 2 else None
            if m.dim() == 2 and r["mismatch_outside_mu"] == 0:
                if name in ("in_proj_a", "in_proj_b", "in_proj_z"):
                    ok = bool((bits(col_pub) == bits(w_mute)).all()) and bool((col_base == 0).all())
                    r["mute_check"] = {"rule": "col[mu] = bf16(-1024/sqrt(hidden)) on all rows",
                                       "match": ok}
                    r["class"] = "mute" if ok else "UNEXPLAINED"
                elif name == "q_proj":
                    nh = pub.shape[0] // (2 * hd)
                    gate = torch.zeros(pub.shape[0], dtype=torch.bool)
                    for h in range(nh):
                        gate[h * 2 * hd + hd:(h + 1) * 2 * hd] = True
                    exp_col = torch.where(gate, w_mute, torch.zeros((), dtype=torch.bfloat16))
                    ok = bool((bits(col_pub) == bits(exp_col)).all())
                    r["mute_check"] = {"rule": "q_proj[gate rows of each head, mu] = mute const; "
                                               "query rows untouched", "match": ok,
                                       "gate_rows": int(gate.sum()), "heads": nh}
                    r["class"] = "mute" if ok else "UNEXPLAINED"
                elif name == "k_proj":
                    nkv = pub.shape[0] // hd
                    key = (col_pub.float() * normed_ballast).view(nkv, hd)
                    rot0 = bool((col_pub.view(nkv, hd)[:, :ROTARY] == 0).all())
                    norms = key[:, ROTARY:].norm(dim=1).tolist()
                    nz = int((col_pub != 0).sum())
                    ok = rot0 and all(abs(x - SINK_NORM) < 0.05 for x in norms) and \
                        layer in full_layers
                    sinks[layer] = col_pub.clone()
                    r["anti_sink_check"] = {
                        "rule": "k_proj[:, mu] = anti_sink/sqrt(hidden); RoPE dims 0..63 of every "
                                "kv head zero; |key[64:]| = 16 per kv head (fitted, not derivable)",
                        "rotary_dims_zero": rot0, "kv_heads": nkv, "per_head_norm": norms,
                        "nonzero": nz, "match": ok}
                    r["class"] = "anti_sink" if ok else "UNEXPLAINED"
        rows.append(r)
        msg = r.get("class", r.get("error"))
        if msg != "deterministic":
            log(f"{pk}: {msg} mism={r.get('mismatch')} where={r.get('where')}")
        del w, exp, pub

    # ---- the mu column must be zero everywhere else on the Qwen side ----------------------
    # (already covered: every other tensor is bit-exact to its zero-padded base)

    # ---- final norm ----------------------------------------------------------------------
    wb = B.get(SRC_TEXT + "norm.weight")
    g = ((1.0 + wb.float()) * scale - 1.0).to(torch.bfloat16)
    pn = P.get(norm_pub)
    m, st = diff_stats(g, pn[:d0])
    nr = {"key": norm_pub, "base_key": SRC_TEXT + "norm.weight", "rule": "wide_norm_rescale [0:d0]",
          "shape": list(pn.shape), **st, "class": "deterministic" if st["equal"] else "UNEXPLAINED",
          "chip_part_note": "norm[d0:] = chip gains (code/tau/id/-1), audited bit-exact in "
                            "stage3-a-rebuild"}
    if not st["equal"]:
        nr["where"] = localize(m)
    # how much would the un-rescaled base norm differ? (shows the rescale is really applied)
    nr["sanity_unrescaled_mismatch"] = int((bits(wb) != bits(pn[:d0])).sum())
    log(f"final norm [0:d0]: {nr['class']} (unrescaled base would mismatch {nr['sanity_unrescaled_mismatch']})")

    # ---- embedding -------------------------------------------------------------------------
    sb, sp = B.slice(SRC_TEXT + "embed_tokens.weight"), P.slice(emb_pub)
    eshape = sp.get_shape()
    vocab = eshape[0]
    chip_vocab = meta["chip_vocab"]
    emb = {"key": emb_pub, "shape": eshape, "base_shape": sb.get_shape()}
    mism, maxd, nchunks = 0, 0.0, 0
    bad_rows = []
    for a in range(0, v0, args.chunk):
        b = min(v0, a + args.chunk)
        eb, ep = sb[a:b, :d0], sp[a:b, :d0]
        m = bits(eb) != bits(ep)
        n = int(m.sum())
        if n:
            mism += n
            maxd = max(maxd, float((eb.float() - ep.float()).abs()[m].max()))
            bad_rows += (m.any(1).nonzero().flatten() + a).tolist()[:16]
        nchunks += 1
    emb["text_rows_qwen_block"] = {"rows": [0, v0], "cols": [0, d0], "numel": v0 * d0,
                                   "mismatch": mism, "equal": mism == 0, "max_abs_diff": maxd,
                                   "bad_rows_sample": bad_rows[:16]}
    log(f"embed[0:{v0}, 0:{d0}] vs base: mismatch {mism}")
    # chip-token rows and pad rows in the Qwen columns
    nz_chip, nz_pad_all = 0, 0
    for a in range(v0, v0 + chip_vocab, args.chunk):
        b = min(v0 + chip_vocab, a + args.chunk)
        nz_chip += int((sp[a:b, :d0] != 0).sum())
    pad = sp[v0 + chip_vocab:vocab, :]
    nz_pad_all = int((pad != 0).sum())
    emb["chip_rows_qwen_cols"] = {"rows": [v0, v0 + chip_vocab], "cols": [0, d0],
                                  "nonzero": nz_chip,
                                  "note": "chip tokens have no Qwen part (expected all zero)"}
    emb["pad_rows"] = {"rows": [v0 + chip_vocab, vocab], "cols": [0, hidden],
                       "nonzero": nz_pad_all, "all_zero": nz_pad_all == 0}
    emb["text_rows_chip_cols_note"] = ("embed[0:v0, d0:] = per-token chip features; audited "
                                       "bit-exact against the compiler in stage3-a-rebuild")
    log(f"chip rows Qwen cols nonzero {nz_chip}; pad rows {vocab - v0 - chip_vocab} nonzero {nz_pad_all}")

    # ---- summary -------------------------------------------------------------------------
    by_class = {}
    for r in rows:
        by_class[r.get("class", "error")] = by_class.get(r.get("class", "error"), 0) + 1
    groups = {}
    for r in rows:
        name = re.sub(r"^model\.layers\.\d+\.", "", r["key"])
        gk = (r["type"], name)
        gg = groups.setdefault(gk, {"type": r["type"], "tensor": name, "rule": r["rule"], "n": 0,
                                    "bit_exact": 0, "mute": 0, "anti_sink": 0, "unexplained": 0,
                                    "mismatch_elems": 0})
        gg["n"] += 1
        c = r.get("class")
        gg["bit_exact" if c == "deterministic" else c.lower() if c in ("mute", "anti_sink")
           else "unexplained"] += 1
        gg["mismatch_elems"] += r.get("mismatch", 0)
    mute_elems = sum(r.get("mismatch", 0) for r in rows if r.get("class") == "mute")
    sink_elems_nz = sum(r.get("mismatch", 0) for r in rows if r.get("class") == "anti_sink")
    sink_col_elems = sum(r["shape"][0] for r in rows if r.get("class") == "anti_sink")
    unexplained = [r["key"] for r in rows if r.get("class") not in ("deterministic", "mute", "anti_sink")]
    qwen_side = len(rows) + 2  # + final norm (Qwen part) + embedding (Qwen block)
    full_exact = by_class.get("deterministic", 0) + int(nr["equal"]) + int(emb["text_rows_qwen_block"]["equal"])
    report["summary"] = {
        "published_tensors": len(pub_keys),
        "chip_layer_tensors_(audited_in_stage3a)": len(chip_keys),
        "qwen_side_tensors": qwen_side,
        "qwen_side_bit_exact_to_base_plus_widen": full_exact,
        "qwen_side_mute_only": by_class.get("mute", 0),
        "qwen_side_anti_sink": by_class.get("anti_sink", 0),
        "unexplained_tensors": unexplained,
        "by_class": by_class,
        "deterministic_mute_delta": {"elements": mute_elems, "bytes_bf16": 2 * mute_elems,
                                     "note": "one constant bf16(-1024/sqrt(4608)), derivable"},
        "non_derivable_residual": {"anti_sink_layers": sorted(sinks), "column_elems": sink_col_elems,
                                   "nonzero_elems": sink_elems_nz, "bytes_bf16": 2 * sink_col_elems,
                                   "bytes_bf16_nonzero_only": 2 * sink_elems_nz},
        "anti_sink_only_on_full_attention_k_proj":
            sorted(sinks) == full_layers and all(
                r["key"].endswith("self_attn.k_proj.weight") for r in rows if r.get("class") == "anti_sink"),
        "full_attention_layers": full_layers,
    }
    report["groups"] = sorted(groups.values(), key=lambda g: (g["type"], g["tensor"]))
    report["final_norm"] = nr
    report["embedding"] = emb
    report["tensors"] = rows
    report["anti_sink_columns_bf16_hex"] = {
        str(k): bits(v).to(torch.int32).bitwise_and(0xFFFF).tolist() for k, v in sorted(sinks.items())}
    report["anti_sink_keys_fp32"] = {str(k): (v.float() * normed_ballast).tolist()
                                     for k, v in sorted(sinks.items())}
    report["runtime"] = {"seconds": round(time.time() - T0, 1), **rss_gb(),
                         "torch": torch.__version__}
    Path(args.out).write_text(json.dumps(report, indent=1))
    log("summary", json.dumps(report["summary"], indent=1))
    log("wrote", args.out, report["runtime"])


if __name__ == "__main__":
    main()
