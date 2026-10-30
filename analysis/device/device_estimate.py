"""Device-tree prototype measurement (Qwenputer stage-3 idea B).

Builds the ADD8 and SORT4 devices (device_graphs.py) in the q35 IR, checks exactness, emits the
Qwen3.5 layers and counts nonzeros the same way as chipformer.rv.measure.nonzeros, optionally
builds the standalone HF model (q35.tiny.build) and checks its argmax, and prints a JSON report.

    repro/qwenputer/experimental/riscv/.venv/Scripts/python.exe analysis/device/device_estimate.py
        [--hf] [--out analysis/device/device_estimate.json]
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro/qwenputer/experimental/riscv"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import device_graphs as D  # noqa: E402
from chipformer.q35 import ir  # noqa: E402
from chipformer.q35.emit import emit_layers  # noqa: E402
from chipformer.q35.ir import FFN, Head  # noqa: E402
from chipformer.q35.tiny import BASE, pick_layout  # noqa: E402

# reference figures (experimental/riscv/docs/rv32.md, docs/q35.md)
RV32 = {"layers": "11-12", "nonzeros": 1_040_595, "dims": 2679, "heads": 99, "ffn_units": 42012,
        "tokens_per_instr": "1.3-1.8", "calc_call_tokens": "254-684 (add 4+3 .. mul 12x12)"}


def predicted(c, tokens, kernel=False):
    m = ir.evaluate(c, tokens, check=True, kernel=kernel)
    codes = np.stack([m[:, next(iter(e.terms))] for e in c.code], 1)
    return [c.decode_out(r) for r in codes]


def check_records(c, records, chunk=512, kernel_records=0):
    """Teacher-forced: at every owned position the device's next token equals the record's."""
    n_pred = n_bad = 0
    first_bad = None
    for s in range(0, len(records), chunk):
        recs = records[s:s + chunk]
        toks = [t for r in recs for t in r]
        pred = predicted(c, toks)
        for p in range(len(toks) - 1):
            if pred[p] is not None:
                n_pred += 1
                if pred[p] != toks[p + 1]:
                    n_bad += 1
                    first_bad = first_bad or (p, toks[p + 1], pred[p])
        # a record ending in an owned position would predict past the end; none do here
    kern = 0
    for r in records[:kernel_records]:  # replay head reads as the HF torch kernels compute them
        predicted(c, r, kernel=True)
        kern += 1
    return {"records": len(records), "owned_predictions": n_pred, "mismatches": n_bad,
            "first_mismatch": first_bad, "kernel_checked_records": kern}


def nonzeros(c):
    """As chipformer.rv.measure.nonzeros: emit_layers on the standalone layout, count != 0."""
    lay = pick_layout(len(c.dims))
    tc = dict(BASE, hidden_size=lay.hidden, linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate)
    sd = emit_layers(c, lay, tc, 0, "model.")
    per, by_kind = Counter(), Counter()
    for k, t in sd.items():
        nz = int((t != 0).sum())
        per[int(k.split(".")[2])] += nz
        kind = ("const(dt_bias,norm)" if k.endswith(("dt_bias", "norm.weight")) else
                "mlp" if ".mlp." in k else "attn")
        by_kind[kind] += nz
    return dict(sorted(per.items())), dict(by_kind), lay.hidden


def graph_term_count(c):
    """Analytical cross-check: nonzeros implied by the graph (Expr terms per projection row)."""
    n = 0
    for node in c.nodes:
        if isinstance(node, FFN):
            for _, a, b in node.terms:
                n += len(a.terms) + len(b.terms) + 1  # gate row, up row, one down weight
        else:
            for vh in node.vheads():
                chans = list(node.q.values()) + list(node.k.values()) + list(vh.v.values())
                n += sum(len(ch.expr.terms) + sum(1 for w in ch.taps if w) for ch in chans)
                n += len(vh.a.terms) + sum(len(z.terms) for z in vh.z.values()) + len(vh.out)
                n += 1 if vh.beta is None else len(vh.beta.terms)
    return n


def per_sublayer(c):
    units, heads = Counter(), Counter()
    for node in c.nodes:
        if isinstance(node, FFN):
            units[node.level] += len(node.terms)
        else:
            heads[node.level] += 1
    return dict(sorted(units.items())), dict(sorted(heads.items()))


def hf_check(c, records, vocab_size):
    """Standalone Qwen3.5 checkpoint (q35.tiny.build, slots shared, exit clears): argmax at every
    owned position of the records."""
    import torch

    from chipformer.q35 import tiny

    model = tiny.build(c, list(range(vocab_size)), c.keep, dtype=torch.bfloat16, share=True)
    total, bad = 0, 0
    with torch.no_grad():
        for r in records:
            ids = torch.tensor([r])
            logits = model(input_ids=ids).logits[0].float()
            pred = predicted(c, r)
            for p in range(len(r) - 1):
                if pred[p] is not None:
                    total += 1
                    bad += int(int(logits[p, :vocab_size].argmax()) != r[p + 1])
    n_params_nz = sum(int((p != 0).sum()) for p in model.parameters())
    return {"records": len(records), "owned_positions": total, "argmax_mismatches": bad,
            "hidden": model.layout.hidden, "model_nonzeros_incl_embedding": n_params_nz}


def describe(name, c, check, tokens_per_op, extra=None):
    nz, by_kind, hidden = nonzeros(c)
    units, heads = per_sublayer(c)
    s = c.stats()
    return {"device": name, **s, "units_per_sublayer": units, "heads_per_sublayer": heads,
            "nonzeros_per_layer": nz, "nonzeros_total": sum(nz.values()),
            "nonzeros_by_kind": by_kind, "graph_term_estimate": graph_term_count(c),
            "standalone_hidden": hidden, "tokens_per_op": tokens_per_op, "exactness": check,
            **(extra or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true", help="also build the HF model and check argmax")
    ap.add_argument("--sort-cases", type=int, default=3000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent /
                                         "device_estimate.json"))
    args = ap.parse_args()
    t0 = time.time()
    rep = {"reference_rv32": RV32}

    add = D.build_add8()
    add_recs = [D.add8_record(a, b) for a in range(256) for b in range(256)]  # exhaustive
    add_chk = check_records(add, add_recs, chunk=2048, kernel_records=8)
    rep["add8"] = describe("ADD8 (8+8 -> 9 bit)", add, add_chk,
                           {"record": "A(a) B(b) S(sum)", "tokens": 3, "device_generated": 1,
                            "host_or_qwen_supplied": 2})

    srt = D.build_sort4()
    rng = random.Random(0)
    edge = [[0, 0, 0, 0], [255] * 4, [1, 2, 3, 4], [4, 3, 2, 1], [7, 7, 3, 3], [0, 255, 0, 255],
            [255, 0, 128, 128], [9, 3, 200, 3]]
    cases = edge + [[rng.randrange(256) for _ in range(4)] for _ in range(args.sort_cases)]
    cases += [[rng.randrange(4) for _ in range(4)] for _ in range(500)]  # many ties
    srt_recs = [D.sort4_record(x) for x in cases]
    srt_chk = check_records(srt, srt_recs, chunk=512, kernel_records=8)
    rep["sort4"] = describe("SORT4 (4 x 8-bit, stable)", srt, srt_chk,
                            {"record": "X0 X1 X2 X3 Y0 Y1 Y2 Y3", "tokens": 8,
                             "device_generated": 4, "host_or_qwen_supplied": 4})

    dad = D.build_dadd()
    pairs = [(x, y) for x in range(100) for y in range(100)]  # every 2-digit case, all carries
    pairs += [(rng.randrange(10**n), rng.randrange(10**n)) for n in (4, 8, 12, 24, 40)
              for _ in range(200)]
    pairs += [(10**n - 1, 1) for n in range(1, 40)]  # long carry chains
    dad_recs = [D.dadd_record(x, y) for x, y in pairs]
    assert all(D.dadd_value(r) == x + y for r, (x, y) in zip(dad_recs, pairs))
    dad_chk = check_records(dad, dad_recs, chunk=1024, kernel_records=8)
    rep["dadd"] = describe("DADD (decimal digit-serial bignum add)", dad, dad_chk,
                           {"record": "G, then per digit: P(a_k,b_k) S(z_k)", "tokens": "2n + 1",
                            "device_generated_per_digit": 1,
                            "add_4+3_digits": {"n": 4, "tokens": 9},
                            "add_12+12_digits": {"n": 12, "tokens": 25}})

    # width scan of the adder (cost growth)
    rep["add_width_scan"] = {}
    for w in (1, 2, 4, 8):
        cw = D.build_add8(w)
        nz, _, _ = nonzeros(cw)
        rep["add_width_scan"][w] = {"layers": cw.n_layers(), "ffn_units": cw.stats()["ffn_units"],
                                    "dims": len(cw.dims), "nonzeros": sum(nz.values())}

    if args.hf:
        rep["add8"]["hf_standalone"] = hf_check(add, add_recs[::997][:40], 1024)
        rep["sort4"]["hf_standalone"] = hf_check(srt, srt_recs[:40], 2048)
        rep["dadd"]["hf_standalone"] = hf_check(dad, dad_recs[-60:], 121)
    rep["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
