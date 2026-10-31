"""Stage-8 E2: can ADD8/SORT4/DADD share the RV core's existing chip layers?

CPU-only IR/allocator probe. It constructs the same published core as RvChat and rv/merge.py,
copies a device graph into it, then calls the real q35 allocator and, only for a fit, emit_layers.
No Qwen weight tensors are loaded and nothing under .data is written.

    repro/qwenputer/experimental/riscv/.venv/Scripts/python.exe -u analysis/device/merge_probe.py
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
T0 = time.time()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro/qwenputer/experimental/riscv"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import device_graphs as D  # noqa: E402
from chipformer.dsl.graph import Expr  # noqa: E402
from chipformer.q35.emit import Layout, allocate, emit_layers  # noqa: E402
from chipformer.q35.ir import FFN, Head, Target  # noqa: E402
from chipformer.q35.tiny import BASE  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import core  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402
from chipformer.rv.store import Config  # noqa: E402


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rss():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        p = PMC()
        p.cb = ctypes.sizeof(PMC)
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return {"ws_gb": round(p.WorkingSetSize / 2**30, 3),
                "peak_ws_gb": round(p.PeakWorkingSetSize / 2**30, 3),
                "peak_commit_gb": round(p.PeakPagefileUsage / 2**30, 3)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def tokenizer(path):
    T = BR.Text(path)
    vocab = T.tok.get_vocab(with_added_tokens=True)
    T.n = vocab.get("<|reserved_0|>", 248320)
    T.bytes = T.bytes[:T.n]
    return T


def usage(g):
    units, heads = Counter(), Counter()
    for n in g.nodes:
        (units if isinstance(n, FFN) else heads)[n.level] += len(n.terms) if isinstance(n, FFN) else 1
    return units, heads


def rebuild(g):
    g.units.clear()
    g.heads.clear()
    for n in g.nodes:
        if isinstance(n, FFN):
            g.units[n.level] = g.units.get(n.level, 0) + len(n.terms)
        else:
            g.heads[n.level] = g.heads.get(n.level, 0) + 1


def profile(g, units_cap, heads_cap):
    units, heads = usage(g)
    rows = []
    for level in range(1, max([*units, *heads], default=0) + 1):
        odd = level % 2 == 1
        used, cap = (heads[level], heads_cap) if odd else (units[level], units_cap)
        rows.append({"level": level, "kind": "attn" if odd else "mlp", "used": used,
                     "free": cap - used})
    return {"stats": g.stats(), "sublayers": rows,
            "tightest": sorted(rows, key=lambda r: (r["free"], r["level"]))[:10]}


def clone_host(src):
    g = copy.deepcopy(src)
    for name in ("owner", "keep", "code", "text", "store", "out_fam", "out_pay", "tau", "mu",
                 "is_text", "isTCC", "probe", "n_plain", "tables"):
        if hasattr(src, name):
            setattr(g, name, getattr(src, name))
    return g


def import_device(dst, dev, prefix, fold_exit=False):
    sizes = {'add8': 1024, 'sort4': 2048, 'dadd': 121}
    offset = S.VOCAB + {'add8': 0, 'sort4': 1024, 'dadd': 3072}[prefix]
    size = sizes[prefix]
    active = dst.token_feature(prefix + '.active', lambda t: int(offset <= t < offset + size), full=True)
    active_dim = next(iter(active.terms))
    fmap = {0: 0}
    for old, dim in enumerate(dev.dims):
        if old == 0:
            continue
        dst.dims.append(copy.copy(dim))
        new = len(dst.dims) - 1
        fmap[old] = new
        dst.dims[new].name = f"{prefix}.{dim.name}"
        if dim.kind == "token":
            old_fn = dev.token_fns[old]
            dst.token_fns[new] = lambda t, fn=old_fn: fn(t - offset) if offset <= t < offset + size else 0
            dst.full_fns.add(new)
    copied = []
    for node in dev.nodes:
        n = copy.deepcopy(node)
        n.name = f"{prefix}.{n.name}"
        if isinstance(n, FFN):
            n.terms = [(c, Expr({fmap[d]: w for d, w in a.terms.items()}),
                        Expr({fmap[d]: w for d, w in b.terms.items()})) for c, a, b in n.terms]
            n.out = fmap[n.out]
        else:
            def remap_chan(ch):
                ch = copy.copy(ch)
                ch.expr = Expr({active_dim if d == 0 else fmap[d]: w for d, w in ch.expr.terms.items()})
                return ch

            def remap(e):
                return Expr({active_dim if d == 0 else fmap[d]: w for d, w in e.terms.items()})

            n.q = {i: remap_chan(ch) for i, ch in n.q.items()}
            n.k = {i: remap_chan(ch) for i, ch in n.k.items()}
            n.v = {i: remap_chan(ch) for i, ch in n.v.items()}
            n.a = remap(n.a)
            n.beta = None if n.beta is None else remap(n.beta)
            n.z = {i: remap(e) for i, e in n.z.items()}
            n.out = [(col, fmap[d], w) for col, d, w in n.out]
            for vh in n.vx:
                vh.v = {i: remap_chan(ch) for i, ch in vh.v.items()}
                vh.a = remap(vh.a)
                vh.beta = None if vh.beta is None else remap(vh.beta)
                vh.z = {i: remap(e) for i, e in vh.z.items()}
                vh.out = [(col, fmap[d], w) for col, d, w in vh.out]
        copied.append(n)
    dst.keep = list(dst.keep) + [fmap[next(iter(e.terms))] for e in dev.code]
    return copied, fmap


def defined_dims(node):
    if isinstance(node, FFN):
        return [] if node.into else [node.out]
    return [d for vh in node.vheads() if not vh.into for _, d, _ in vh.out]


def move_node(g, node, level):
    for d in defined_dims(node):
        if g.dims[d].level == node.level:
            g.dims[d].level = level
    node.level = level


def pack(g, prefix, unit_cap, head_cap, fold_exit):
    """Capacity-only import schedule preserving device stage barriers.

    Device exits are renamed exit.* and delayed to the host's final exit. This preserves
    private device code outputs; it intentionally does NOT claim unified tokenizer semantics.
    """
    nodes = [n for n in g.nodes if n.name.startswith(prefix + ".")]
    host = [n for n in g.nodes if n not in nodes]
    top = max(n.level for n in host)
    stages = defaultdict(list)
    for n in nodes:
        stages[n.level].append(n)
    used_u, used_h = Counter(), Counter()
    for n in host:
        if isinstance(n, FFN):
            used_u[n.level] += len(n.terms)
        else:
            used_h[n.level] += 1
    ready = 0
    placements = []
    for old, group in sorted(stages.items()):
        exits = all('.exit.' in n.name for n in group)
        odd = isinstance(group[0], Head)
        need = sum(1 if odd else len(n.terms) for n in group)
        candidates = [top] if exits else range(ready + 1, top)
        found = None
        for level in candidates:
            if level <= ready or (level % 2 == 1) != odd:
                continue
            used, cap = (used_h, head_cap) if odd else (used_u, unit_cap)
            if used[level] + need <= cap:
                found = level
                used[level] += need
                break
        if found is None:
            return {'ok': False, 'stage': old, 'need': need, 'ready_after': ready,
                    'exit': exits, 'host_exit_free': unit_cap - used_u[top]}
        for n in group:
            move_node(g, n, found)
            if exits:
                n.name = 'exit.' + n.name
        ready = found
        placements.append({'original_level': old, 'level': found, 'need': need})
    rebuild(g)
    return {'ok': True, 'placements': placements}


def allocate_summary(g, cap):
    try:
        a = allocate(g, None, set(g.keep), cap)
        return {"ok": True, "slots": a.n_slots, "subtract": len(a.subtract),
                "clears": {str(k): len(v) for k, v in sorted(a.clears.items())}, "alloc": a}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


def emit_summary(g, alloc, hidden, m, cap):
    lay = Layout(hidden=hidden, off=2560, m=m, slots=alloc.slots)
    assert lay.off + m + 1 + alloc.n_slots <= hidden
    tc = dict(BASE, hidden_size=hidden, intermediate_size=cap,
              linear_num_key_heads=g.target.key_heads, linear_num_value_heads=g.target.value_heads)
    sd = emit_layers(g, lay, tc, 0, "model.", alloc, set(g.keep))
    per = Counter()
    for key, tensor in sd.items():
        per[int(key.split(".")[2])] += int((tensor != 0).sum())
    out = {"scale": lay.scale, "slots": alloc.n_slots, "nonzeros": sum(per.values()),
           "per_layer": dict(sorted(per.items()))}
    del sd
    return out


def namespace(T):
    return {"rv_vocab": S.VOCAB, "text_marker": S.TEXT_TOKEN, "ordinary_feature_mask": 65535,
            "add8_ids_that_alias_rv": [t for t in range(D.S_BASE + 512) if t < S.VOCAB][:8],
            "alias_count": min(D.S_BASE + 512, S.VOCAB),
            "candidate_above_text_marker_aliases_to": (S.TEXT_TOKEN + T.n) & 65535,
            "recovered_qwen_vocab": T.n}


def run_case(base, dev, name, mode, caps, emit, hidden, ballast):
    g = clone_host(base)
    copied, _ = import_device(g, dev, name, mode == "folded")
    exit_at = next(i for i, n in enumerate(g.nodes) if n.name.startswith("exit."))
    ordinary = [n for n in copied if not n.name.startswith(name + ".exit.")]
    exits = [n for n in copied if n.name.startswith(name + ".exit.")]
    g.nodes[exit_at:exit_at] = ordinary
    g.nodes.extend(exits)
    rebuild(g)
    if mode == "native":
        # Retain the device's internal relative placement, starting at level 1. Host levels were
        # already occupied, so this exposes the constructor-level collision directly.
        base_level = min(n.level for n in copied)
        for n in copied:
            move_node(g, n, n.level - base_level + 1)
        rebuild(g)
        packing = {"ok": True, "preserved_relative_levels": True}
    else:
        packing = pack(g, name, caps[0], caps[1], mode == "folded")
    prof = profile(g, caps[0], caps[1])
    alloc = allocate_summary(g, caps[0]) if packing["ok"] else {"ok": False, "skipped": True}
    fit = bool(packing["ok"] and alloc.get("ok") and g.n_layers() == base.n_layers()
               and prof["stats"]["max_units"] <= caps[0] and prof["stats"]["max_heads"] <= caps[1])
    row = {"mode": mode, "device": name, "packing": {k: v for k, v in packing.items()},
           "stats": prof["stats"], "tightest": prof["tightest"],
           "allocator": {k: v for k, v in alloc.items() if k != "alloc"},
           "fits_without_extra_layers": fit}
    row['schedule'] = [{'name': n.name, 'level': n.level} for n in g.nodes
                       if n.name.startswith(name + '.') or n.name.startswith('exit.' + name + '.')]
    if alloc.get('ok'):
        a = alloc['alloc']
        actual = Counter(g.units)
        for n in g.nodes:
            if isinstance(n, FFN) and n.out in a.subtract:
                actual[n.level] += 1
        for lv, ds in a.clears.items():
            actual[lv] += len(ds)
        actual[max(n.level for n in g.nodes)] += sum(d not in set(g.keep) for d in a.final) + ballast
        row['emitted_units_including_clears_ballast'] = dict(sorted(actual.items()))
    if emit and fit:
        try:
            row['emit'] = emit_summary(g, alloc['alloc'], hidden, ballast, caps[0])
        except Exception as e:
            row['emit'] = {'ok': False, 'error': repr(e), 'trace': traceback.format_exc()}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / ".data/Qwen3.5-4B-Qwenputer-Filesystem"))
    ap.add_argument("--devices", default="add8")
    ap.add_argument("--modes", default="packed", help="packed or native (diagnostic only)")
    ap.add_argument("--emit-fit", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "merge_probe.json"))
    args = ap.parse_args()
    model = Path(args.model)
    meta = json.loads((model / "chipformer.json").read_text(encoding="utf-8"))
    tc = json.loads((model / "config.json").read_text(encoding="utf-8"))
    target = Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["intermediate_size"])
    T = tokenizer(model / "tokenizer.json")
    store = Config(**meta["store"])
    log(f"build core {target}")
    base = core.build(target, T, store)
    base_alloc = allocate_summary(base, target.intermediate)
    caps = (target.intermediate, target.key_heads)
    report = {"baseline": {**profile(base, *caps), "published": meta["circuit"] | {"slots": meta["slots"]},
                           "allocator": {k: v for k, v in base_alloc.items() if k != "alloc"},
                           "memory": rss()},
              "namespace": namespace(T), "devices": {}, "experiments": []}
    log(f"baseline layers={base.n_layers()} slots={base_alloc.get('slots')} {rss()}")
    builders = {"add8": D.build_add8, "sort4": D.build_sort4, "dadd": D.build_dadd}
    devs = {name: builders[name]() for name in args.devices.split(",") if name}
    for name, dev in devs.items():
        report["devices"][name] = {**dev.stats(), "units": dict(usage(dev)[0]),
                                   "heads": dict(usage(dev)[1])}
        log(f"device {name}: {report['devices'][name]}")
    if args.emit_fit:
        log('emitting baseline on CPU')
        report['baseline']['emit'] = emit_summary(base, base_alloc['alloc'], meta['hidden'], meta['ballast_dims'], caps[0])
        log(str(report['baseline']['emit']))
    for mode in args.modes.split(","):
        for name, dev in devs.items():
            row = run_case(base, dev, name, mode, caps, args.emit_fit, meta["hidden"], meta["ballast_dims"])
            report["experiments"].append(row)
            log(f"{mode}/{name}: fit={row['fits_without_extra_layers']} stats={row['stats']} "
                f"alloc={row['allocator']}")
    combined = clone_host(base)
    for name, dev in devs.items():
        copied, mapping = import_device(combined, dev, name, False)
        dev.merge_mapping = mapping
        combined.nodes.extend(copied)
        result = pack(combined, name, caps[0], caps[1], False)
        assert result['ok'], result
    rebuild(combined)
    from chipformer.q35 import ir
    replay = {}
    for name, dev in devs.items():
        rec = {'add8': lambda: D.add8_record(255, 1),
               'sort4': lambda: D.sort4_record([255, 0, 127, 127]),
               'dadd': lambda: D.dadd_record(999, 1)}[name]()
        offset = S.VOCAB + {'add8': 0, 'sort4': 1024, 'dadd': 3072}[name]
        try:
            reference = ir.evaluate(dev, rec, check=True)
            merged = ir.evaluate(combined, [offset + t for t in rec], check=True)
            columns = [next(iter(e.terms)) for e in dev.code]
            equal = bool((reference[:, columns] == merged[:, [dev.merge_mapping[d] for d in columns]]).all())
            replay[name] = {'exact': True, 'private_code_equal': equal, 'tokens': len(rec),
                            'host_owner_values': ir.ev(combined.owner, merged).tolist()}
        except Exception as e:
            replay[name] = {'exact': False, 'error': repr(e)}
    report['semantic_replay'] = replay
    ca = allocate_summary(combined, caps[0])
    report['combined'] = {**profile(combined, *caps), 'allocator': {k:v for k,v in ca.items() if k != 'alloc'}}
    if args.emit_fit:
        log('emitting combined devices on CPU')
        try:
            report['combined']['emit'] = emit_summary(combined, ca['alloc'], meta['hidden'], meta['ballast_dims'], caps[0])
        except Exception as e:
            report['combined']['emit'] = {'ok': False, 'error': repr(e), 'trace': traceback.format_exc()}
    if args.tiny:
        tt = Target(16, 16, 3584)
        tiny = core.build(tt, None, None)
        ta = allocate_summary(tiny, tt.intermediate)
        report["tiny"] = {"baseline": {**profile(tiny, tt.intermediate, tt.key_heads),
                                       "allocator": {k: v for k, v in ta.items() if k != "alloc"}},
                          "experiments": []}
        for name, dev in devs.items():
            row = run_case(tiny, dev, name, "folded", (tt.intermediate, tt.key_heads), False, 0, 0)
            report["tiny"]["experiments"].append(row)
            log(f"tiny/{name}: fit={row['fits_without_extra_layers']} stats={row['stats']}")
    report["memory_final"] = rss()
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()