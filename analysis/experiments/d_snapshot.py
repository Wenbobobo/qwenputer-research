#!/usr/bin/env python
"""Stage-3 idea D: what a Qwenputer store snapshot is bound to, and how big it is as a function
of content.

Reference store only (chipformer/rv/store.py, the protocol the circuit is checked against). No
model weights are loaded. Run with the riscv venv:

    <riscv>/.venv/Scripts/python.exe -I analysis/experiments/d_snapshot.py          (parts 1 and 2)
    <riscv>/.venv/Scripts/python.exe -I analysis/experiments/d_snapshot.py --part3  (circuit probe)

RISCV_DIR overrides the riscv checkout. Results go to analysis/experiments/d_snapshot_results.json.
The circuit probe runs in child processes under a 1900 MB working-set watchdog.
"""

import ctypes
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

RISCV = Path(os.environ.get("RISCV_DIR", "C:/Projects/Qcompute/repro/qwenputer/experimental/riscv"))
TOKENIZER_FALLBACK = Path("C:/Users/Wenbo/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/"
                          "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/tokenizer.json")
CALC = RISCV / "tests" / "data" / "rv" / "calc-bridge.elf"
RAM_LIMIT_MB = 1900
OUT = Path(__file__).resolve().parent / "d_snapshot_results.json"
sys.path.insert(0, str(RISCV))

from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402
from chipformer.rv import store as ST  # noqa: E402


# ---------------------------------------------------------------- memory watchdog (Windows)


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def mem_mb():
    """(current, peak) working set of this process in MB; (None, None) off Windows."""
    if os.name != "nt":
        return None, None
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20


def start_watchdog(limit_mb):
    def run():
        while True:
            cur, _ = mem_mb()
            if cur is not None and cur > limit_mb:
                print(f"WATCHDOG: working set {cur:.0f} MB > {limit_mb} MB, aborting",
                      file=sys.stderr, flush=True)
                os._exit(5)
            time.sleep(0.2)
    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- helpers


def load_tokenizer():
    try:
        path, src = BR.default_tokenizer(), "HF cache Qwen3.5 tokenizer"
    except (FileNotFoundError, RuntimeError):
        path = TOKENIZER_FALLBACK
        src = ("Qwen3.6-27B tokenizer.json: no Qwen3.5 tokenizer in the HF cache; "
               "the store's hard-coded token ids are checked against it (check_constants)")
    return BR.Text(path), str(path), src


def check_constants(T):
    """The store's hard-coded Qwen ids against this tokenizer (a subset of tests' test_token_ids)."""
    want = {"\n": [ST.Q_NL], "=": [ST.Q_EQ], "function": [ST.Q_FUNCTION],
            "parameter": [ST.Q_PARAMETER], "</": [ST.Q_LTSLASH], "<": [ST.Q_LT], ">": [ST.Q_GT]}
    for op, q in ST.OP_BARE.items():
        want[op] = [q]
    for p, q in ST.PARAM_BARE.items():
        want[p] = [q]
    bad = []
    for s, w in want.items():
        got = T.encode(s)
        if got != w:
            bad.append([repr(s), got, w])
    vocab = T.tok.get_vocab(with_added_tokens=True)
    for s, q in (("<tool_call>", ST.Q_TCO), ("</tool_call>", ST.Q_TCC), ("<|endoftext|>", ST.SPECIAL)):
        if vocab.get(s) != q:
            bad.append([s, vocab.get(s), q])
    return {"checked": len(want) + 3, "vocab_size": len(vocab), "mismatches": bad}


def fresh_machine():
    m = ref.Machine.from_program(elf.program(str(CALC)))
    BR.park(m)
    return m


def session(T, cfg, dev=None):
    return ST.StoreSession(T, fresh_machine(), cfg, dev)


def do(ses, kind, name=None, text=None):
    """One store call; returns the body the store printed for it."""
    start = len(ses.toks)
    ses.call(kind, name, text)
    out = ses.printed(start)
    return out[0] if len(out) == 1 else out


def logical_bits(dev, cfg):
    """Bits of store.py's Device if packed: directory, order, lens, nexts, content, registers."""
    cb = (cfg.n_chunks - 1).bit_length()  # chunk index (9 for 512)
    lb = (ST.CMAX).bit_length()  # text length (7 for 104)
    pb = (cfg.n_chunks * ST.CMAX).bit_length()  # content position chunk*CMAX+index (16 for 512)
    name_bits = ST.LEN_BITS + ST.NMAX * ST.ID_BITS  # length + 6 token ids = 111
    parts = {
        "dir (slot, name key, alive, head, tail)": len(dev.dir) * (cfg.slot_bits + name_bits + 1 + 2 * cb),
        "order (slot per name)": len(dev.order) * cfg.slot_bits,
        "lens (chunk -> length)": len(dev.lens) * (cb + lb),
        "nexts (chunk -> next chunk)": len(dev.nexts) * 2 * cb,
        "content ((chunk,index) -> token id)": len(dev.content) * (pb + ST.ID_BITS),
        "registers (NEXTG, GCUR, LCUR, NORD)": 3 * cb + 10,
    }
    return {k: math.ceil(v / 8) for k, v in parts.items()}, math.ceil(sum(parts.values()) / 8)


def touched_rows(dev):
    crow = {(g, i // ST.TPR) for (g, i) in dev.content}
    return {"dir": len(dev.dir), "order": len(dev.order), "len": len(dev.lens),
            "next": len(dev.nexts), "content": len(crow),
            "regs": sum(1 for v in (dev.nextg, dev.gcur, dev.lcur) if v) + (1 if dev.order else 0)}


def capacity_rows(cfg):
    return {"dir": cfg.n_slots, "order": cfg.n_names, "len": cfg.n_chunks,
            "next": cfg.n_chunks, "content": cfg.n_chunks * ST.ROWS_PER_CHUNK, "regs": 4}


def name_bytes(i):
    return f"file {i:03d}"


def text_str(i):
    return f"locker code is {4471 + i}-AZ"


# ---------------------------------------------------------------- part 1: size vs files


def measure(T, n):
    cfg = ST.Config(seed=2035)
    ses = session(T, cfg)
    names_s = [name_bytes(i) for i in range(n)]
    texts_s = [text_str(i) for i in range(n)]
    names = [T.encode(s) for s in names_s]
    texts = [T.encode(s) for s in texts_s]
    t0 = time.time()
    oks = sum(do(ses, "put", nm, tx) == "ok\n" for nm, tx in zip(names, texts))
    secs = time.time() - t0
    dev = ses.dev
    parts, lbytes = logical_bits(dev, cfg)
    rows = touched_rows(dev)
    text_bytes = sum(len(s.encode()) for s in names_s + texts_s)
    tok_count = sum(len(t) for t in names + texts)
    out = {
        "files": n, "puts_ok": oks, "names_in_store": len(dev.names()), "seconds": round(secs, 2),
        "logical_bytes": lbytes, "logical_parts_bytes": parts,
        "pickle_bytes_of_Device": len(pickle.dumps(dev, protocol=5)),
        "touched_rows": rows, "touched_rows_total": sum(rows.values()),
        "capacity_rows": capacity_rows(cfg), "capacity_rows_total": sum(capacity_rows(cfg).values()),
        "text_utf8_bytes": text_bytes, "token_ids": tok_count,
        "token_ids_as_u32_bytes": 4 * tok_count,
        "ratio_logical_over_text": round(lbytes / text_bytes, 3),
        "ratio_logical_over_text_u32": round(lbytes / (4 * tok_count), 3),
        "logical_bytes_per_file": round(lbytes / max(n, 1), 1),
    }
    if n == 512:
        out["extra_put_when_full"] = do(ses, "put", T.encode("extra fact"), T.encode("x"))
    return out, ses, names, texts_s


# ---------------------------------------------------------------- part 2: portability


def slot_agreement(cfg, dev):
    hits = sum(1 for slot, e in dev.dir.items() if cfg.slot_of(e.key) == slot)
    return hits, len(dev.dir)


def lookup(T, cfg, blob, names, texts_s):
    dev = pickle.loads(blob)
    ses = session(T, cfg, dev)
    tally = {"success": 0, "not_found": 0, "other": 0}
    ex = []
    for nm, s in zip(names, texts_s):
        got = do(ses, "get", nm)
        g = got if isinstance(got, str) else repr(got)
        if g.strip() == s.strip():
            tally["success"] += 1
        elif "not found" in g:
            tally["not_found"] += 1
        else:
            tally["other"] += 1
        if len(ex) < 2:
            ex.append(g[:80])
    return tally, ex


def long_names(T):
    """Names longer than NMAX tokens: the store keeps the first NMAX tokens and the length
    (capped at NMAX + 1), so names sharing that prefix are one key."""
    base = T.encode("the locker code for the safe room in the old house")
    if len(base) < 8 or base[6] == base[7]:
        return {"skipped": "need 8 tokens with base[6] != base[7]", "tokens": len(base)}
    A = base[:8]
    B = base[:6] + [base[7], base[6]]  # same first 6 tokens, same capped length, other tail
    ses = session(T, ST.Config())
    r = {"tokens_in_sentence": len(base),
         "A_name": T.decode_bytes(A).decode("utf-8", "replace"),
         "B_name": T.decode_bytes(B).decode("utf-8", "replace")}
    r["put_A"] = do(ses, "put", A, T.encode("value A"))
    r["get_A_before_B"] = do(ses, "get", A)
    r["put_B"] = do(ses, "put", B, T.encode("value B"))
    r["get_A_after_B"] = do(ses, "get", A)
    r["get_B"] = do(ses, "get", B)
    r["list"] = do(ses, "list")
    r["names_in_store"] = len(ses.dev.names())
    return r


def part12():
    T, tok_path, tok_src = load_tokenizer()
    res = {"meta": {"riscv_dir": str(RISCV), "tokenizer_path": tok_path, "tokenizer_source": tok_src,
                    "store_constants": ST.__name__, "NMAX": ST.NMAX, "CMAX": ST.CMAX, "TPR": ST.TPR,
                    "rows_per_chunk": ST.ROWS_PER_CHUNK, "HD_state_row_len": 128}}
    res["constants"] = check_constants(T)
    sizes, keep = {}, {}
    for n in (1, 10, 100, 512):
        out, ses, names, texts = measure(T, n)
        sizes[str(n)] = out
        if n == 10:
            keep = {"blob": pickle.dumps(ses.dev, protocol=5), "names": names, "texts": texts}
    res["size_vs_files"] = sizes
    blob, names, texts = keep["blob"], keep["names"], keep["texts"]
    variants = [
        ("same seed 2035", ST.Config(seed=2035)),
        ("seed 2036", ST.Config(seed=2036)),
        ("seed 2037", ST.Config(seed=2037)),
        ("seed 2038", ST.Config(seed=2038)),
        ("slot_bits 8 (n_names 128), seed 2035", ST.Config(slot_bits=8, n_names=128, seed=2035)),
    ]
    port = {}
    for label, cfg_v in variants:
        try:
            tally, ex = lookup(T, cfg_v, blob, names, texts)
            hits, total = slot_agreement(cfg_v, pickle.loads(blob))
            port[label] = {**tally, "success_rate": round(tally["success"] / len(names), 3),
                           "slot_agreement": f"{hits}/{total}", "examples": ex}
        except Exception as e:  # recorded, not hidden
            port[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    res["portability"] = port
    # the restored store keeps working under the same seed (new put after restore)
    ses = session(T, ST.Config(seed=2035), pickle.loads(blob))
    res["restored_same_seed_new_put"] = {
        "put": do(ses, "put", T.encode("file new"), T.encode("new fact")),
        "get": do(ses, "get", T.encode("file new"))}
    res["long_names"] = long_names(T)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


# ---------------------------------------------------------------- part 3: circuit probe (children)


def probe_child(mode):
    from chipformer.q35.ir import HD, Target
    from chipformer.rv import core

    T, tok_path, tok_src = load_tokenizer()
    cfg = ST.Config(slot_bits=4, n_chunks=16, n_names=8, hash_bits=2) if mode == "small" else ST.Config()
    tgt = Target(key_heads=2, value_heads=4, intermediate=64)
    rec = {"mode": mode, "config": repr(cfg), "target": repr(tgt), "HD": HD, "tokenizer": tok_src}
    t0 = time.time()
    try:
        c = core.build(target=tgt, text=T, store=cfg)
    except Exception as e:  # the failure is the result
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")
    else:
        heads = {name: len(node.vheads()) for name, node in c.store.persistent.items()}
        rec.update(ok=True, persistent_vheads=heads, total_vheads=sum(heads.values()),
                   nodes=len(c.nodes))
    rec["seconds"] = round(time.time() - t0, 1)
    cur, peak = mem_mb()
    rec["working_set_mb"] = None if cur is None else round(cur, 1)
    rec["peak_working_set_mb"] = None if peak is None else round(peak, 1)
    return rec


def part3():
    out = {}
    for mode in ("small", "full"):
        try:
            p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()),
                                "--probe-child", mode],
                               capture_output=True, text=True, timeout=420, env=dict(os.environ))
            last = (p.stdout.strip().splitlines() or [""])[-1]
            try:
                out[mode] = json.loads(last)
            except json.JSONDecodeError:
                out[mode] = {"ok": False, "stderr_tail": p.stderr[-600:], "stdout_tail": p.stdout[-300:]}
            out[mode]["exit"] = p.returncode
        except subprocess.TimeoutExpired:
            out[mode] = {"ok": False, "error": "timeout 420 s"}
    full = out.get("full", {})
    if full.get("ok"):
        full["dense_state_bytes_float64"] = full["total_vheads"] * full["HD"] * full["HD"] * 8
        full["dense_state_bytes_float32"] = full["total_vheads"] * full["HD"] * full["HD"] * 4
    return out


def main(argv):
    if len(argv) >= 3 and argv[1] == "--probe-child":
        print(json.dumps(probe_child(argv[2])))
        return
    start_watchdog(RAM_LIMIT_MB)
    if "--part3" in argv:
        res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        res["part3_circuit_probe"] = part3()
        OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(res["part3_circuit_probe"], indent=2, ensure_ascii=False))
        return
    res = part12()
    print("constants mismatches:", res["constants"]["mismatches"])
    print("files | logical B | text B | ratio | pickle B | touched rows | capacity rows")
    for n, o in res["size_vs_files"].items():
        print(n, o["logical_bytes"], o["text_utf8_bytes"], o["ratio_logical_over_text"],
              o["pickle_bytes_of_Device"], o["touched_rows_total"], o["capacity_rows_total"])
    print(json.dumps(res["portability"], indent=1, ensure_ascii=False))
    print(json.dumps(res["restored_same_seed_new_put"], ensure_ascii=False))
    print(json.dumps(res["long_names"], indent=1, ensure_ascii=False))
    cur, peak = mem_mb()
    print(f"working set now {cur:.0f} MB, peak {peak:.0f} MB")


if __name__ == "__main__":
    main(sys.argv)
