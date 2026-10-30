# -*- coding: utf-8 -*-
"""Tests for analysis/memory_disk/memory_disk.py (CPU-only, no model loading).

Run (from C:/Projects/Qcompute):
    $env:PYTHONPATH="C:/Projects/Qcompute/repro/qwenputer/experimental/riscv"
    & .venv/Scripts/python.exe analysis/memory_disk/test_memory_disk.py

Groups:
  1. roundtrip  demo/store.npz -> save -> load -> exact array equality
  2. fingerprint mismatch rejection (FingerprintMismatch / force / verify)
  3. journal replay logical portability (same text across seeds 2035/2036)
  4. NameGuard collision detection on the stage-3 D example + safe short names

A consolidated log is written to analysis/memory_disk/test_log.txt.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memory_disk as MD  # noqa: E402

DEMO = ROOT / "demo" / "store.npz"
MODEL_DIR = ROOT / ".data" / "Qwen3.5-4B-Qwenputer-Filesystem"
EXTENDED_TOK = MODEL_DIR / "tokenizer.json"
RISCV = ROOT / "repro" / "qwenputer" / "experimental" / "riscv"

# the stage-3 D long-name pair (same store key: same first 6 tokens, same length)
NAME_A = "the locker code for the safe room in"
NAME_B = "the locker code for the safe in room"

JOURNAL = [
    {"name": "the locker code for the safe room in", "text": "locker code is 4471-AZ"},
    {"name": "city", "text": "Shanghai, China"},
    {"name": "dog_name", "text": "Biscuit"},
    {"name": "favorite color", "text": "blue"},
    {"name": "pin", "text": "2048"},
    {"name": "a name with six words in it exactly", "text": "six-word name text"},
    {"name": "notes/plan", "text": "Buy milk, then call Bob at 5pm."},
    {"name": "short", "text": "hi"},
    {"name": "empty-ish", "text": ""},
    {"name": "last", "text": "the end"},
]

LOG_LINES: list[str] = []
FAILURES: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    LOG_LINES.append(msg)

def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    log(f"[{tag}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def build_reference_session(seed: int = 2035):
    """A fresh chipformer.rv.store.StoreSession on the extended tokenizer (CPU-only)."""
    T, m, ST = MD._reference_components(EXTENDED_TOK, MD.DEFAULT_MACHINE_ELF)
    cfg = ST.Config(seed=seed)
    return ST.StoreSession(T, m, cfg), T, ST

def store_put(ses, T, name: str, text: str):
    start = len(ses.toks)
    ans = ses.call("put", T.encode(name), T.encode(text))
    printed = ses.printed(start)
    return ans, printed[0] if printed else None

def store_get(ses, T, name: str):
    start = len(ses.toks)
    ses.call("get", T.encode(name))
    printed = ses.printed(start)
    return printed[0] if printed else None

def assert_snapshot_equal(snap_a, snap_b):
    if set(snap_a) != set(snap_b):
        return f"head names differ: {sorted(snap_a)} vs {sorted(snap_b)}"
    for head, vs_a in snap_a.items():
        vs_b = snap_b[head]
        if len(vs_a) != len(vs_b):
            return f"{head}: value-head count {len(vs_a)} != {len(vs_b)}"
        for x, (ra, rb) in enumerate(zip(vs_a, vs_b)):
            if set(ra) != set(rb):
                return f"{head}/{x}: row keys differ: {sorted(ra)} vs {sorted(rb)}"
            for r in ra:
                if not np.array_equal(np.asarray(ra[r]), np.asarray(rb[r])):
                    return f"{head}/{x}/{r}: array values differ"
    return None

# --------------------------------------------------------------------------------------
# 1. roundtrip
# --------------------------------------------------------------------------------------

def test_roundtrip(tmpdir: Path):
    log("\n=== 1. roundtrip demo/store.npz -> save -> load -> exact equality ===")
    snap0 = MD.parse_snapshot_npz(DEMO)
    n_heads = len(snap0)
    n_rows = sum(len(rows) for vs in snap0.values() for rows in vs)
    log(f"demo snapshot: {n_heads} heads, {n_rows} nonzero state rows, "
        f"body {DEMO.stat().st_size} bytes")
    check("demo snapshot parses", n_heads > 0 and n_rows > 0)
    fp = MD.fingerprint(MODEL_DIR)
    fphex = fp["fp"]
    log(f"fingerprint({MODEL_DIR.name}) = {fphex}")
    body = tmpdir / "rt.npz"
    MD.save(snap0, body, fp, journal=JOURNAL)
    header = json.loads(Path(str(body) + ".json").read_text(encoding="utf-8"))
    check("header magic QPUMEM01", header["magic"] == "QPUMEM01")
    check("header format_version 1", header["format_version"] == 1)
    check("header fingerprint equals fp", header["fingerprint"]["fp"] == fp["fp"])
    check("header journal exported", header["journal"] == MD.export_journal(JOURNAL))
    check("header body_sha256 present", len(header["body_sha256"]) == 64)
    snap1 = MD.load(body, expected_fp=fp)
    diff = assert_snapshot_equal(snap0, snap1)
    check("snapshot arrays exactly equal", diff is None, diff or "")
    v = MD.verify(body, expected_fp=fp)
    check("verify booleans all true",
          v["magic_ok"] and v["version_ok"] and v["checksum_ok"] and v["fingerprint_match"])
    hdr = Path(str(body) + ".json")
    log(f"roundtrip body {body.stat().st_size} bytes, header {hdr.stat().st_size} bytes")
    return body

# --------------------------------------------------------------------------------------
# 2. fingerprint mismatch
# --------------------------------------------------------------------------------------

def test_fingerprint_mismatch(tmpdir: Path, body: Path):
    log("\n=== 2. fingerprint mismatch rejection ===")
    fp = MD.fingerprint(MODEL_DIR)
    other = dict(fp)
    other["fp"] = "00" * 32
    try:
        MD.load(body, expected_fp=other)
        raised = False
    except MD.FingerprintMismatch as exc:
        raised = True
        log(f"  FingerprintMismatch raised: {str(exc)[:120]}...")
    check("mismatch raises FingerprintMismatch", raised)
    snap = MD.load(body, expected_fp=other, force=True)
    check("force=True loads anyway", len(snap) > 0)
    snap_ok = MD.load(body, expected_fp=fp)
    check("matching fp loads", len(snap_ok) > 0)
    v_bad = MD.verify(body, expected_fp=other)
    v_good = MD.verify(body, expected_fp=fp)
    check("verify fingerprint_match False/True",
          v_bad["fingerprint_match"] is False and v_good["fingerprint_match"] is True)
    # checksum tamper
    data = bytearray(body.read_bytes())
    data[-1] ^= 0x01
    tampered = tmpdir / "tampered.npz"
    tampered.write_bytes(bytes(data))
    import shutil
    shutil.copyfile(str(body) + ".json", str(tampered) + ".json")
    try:
        MD.load(tampered)
        raised = False
    except MD.MemoryDiskError as exc:
        raised = True
        log(f"  checksum error raised: {str(exc)[:100]}")
    check("tampered body rejected (checksum)", raised)

# --------------------------------------------------------------------------------------
# 3. journal replay portability
# --------------------------------------------------------------------------------------

def test_journal_portability():
    log("\n=== 3. journal replay logical portability ===")
    journal = MD.export_journal(JOURNAL)
    t0 = time.perf_counter()
    s1, r1 = MD.replay_into_reference_store(journal, seed=2035)
    t1 = time.perf_counter()
    checks = [r["answer"] for r in r1]
    check("replay seed 2035: all puts ok", all(c == "ok\n" for c in checks), f"{checks}")
    texts1 = [MD.reference_get(s1, it["name"]) for it in journal]
    expected = [it["text"] + "\n" for it in journal]
    ok1 = texts1 == expected
    check("seed 2035 get() returns original texts", ok1)
    s2, r2 = MD.replay_into_reference_store(journal, seed=2036)
    checks2 = [r["answer"] for r in r2]
    check("replay seed 2036: all puts ok", all(c == "ok\n" for c in checks2))
    texts2 = [MD.reference_get(s2, it["name"]) for it in journal]
    ok2 = texts2 == expected
    check("seed 2036 get() returns original texts (portable)", ok2)
    check("identical texts across seeds", texts1 == texts2)
    # demonstrate the raw state does differ: slot maps for one name across seeds
    T, m, ST = MD._reference_components(EXTENDED_TOK, MD.DEFAULT_MACHINE_ELF)
    key = MD.TokenizerKey(EXTENDED_TOK).key("city")
    slot35 = ST.Config(seed=2035).slot_of(key)
    slot36 = ST.Config(seed=2036).slot_of(key)
    log(f"  state differs but content same: slot(city) seed2035={slot35}, seed2036={slot36}")
    check("different seed -> different slot layout", slot35 != slot36)
    log(f"  reference replay (10 puts + 20 gets) took {t1 - t0:.3f}s")
    return s1, s2

# --------------------------------------------------------------------------------------
# 4. NameGuard
# --------------------------------------------------------------------------------------

def test_nameguard():
    log("\n=== 4. NameGuard collision detection ===")
    guard = MD.NameGuard(EXTENDED_TOK)
    ka = guard.key(NAME_A)
    kb = guard.key(NAME_B)
    log(f"  key({NAME_A!r}) = {ka}")
    log(f"  key({NAME_B!r}) = {kb}")
    check("D example: identical keys (the bug)", ka == kb)
    ca = guard.check(NAME_A)
    check("first name: no collision", ca["collision"] is False and ca["suggestion"] is None)
    guard.register(NAME_A)
    cb = guard.check(NAME_B)
    check("second name: collision detected",
          cb["collision"] is True and cb["existing_name"] == NAME_A)
    alias = cb["suggestion"]
    log(f"  suggested alias for B: {alias!r}")
    check("suggestion has a distinct key", alias is not None and guard.key(alias) != ka)
    # against a real reference store
    ses, T, ST = build_reference_session(seed=2035)
    sc = lambda name, text: store_put(ses, T, name, text)
    guard.put_checked(NAME_A, "value A", store_call=sc)
    try:
        guard.put_checked(NAME_B, "value B", store_call=sc)
        raised = False
    except MD.NameConflict as exc:
        raised = True
        log(f"  NameConflict raised: {str(exc)[:140]}...")
    check("put_checked(B) raises NameConflict", raised)
    check("A still intact after refused B", store_get(ses, T, NAME_A) == "value A\n")
    # B shares A's store key, so get(B) reads A's text (the bug's mechanism);
    # the guard refused the put, so no second file exists and A is unoverwritten.
    check("get(B) sees A's text (same key)", store_get(ses, T, NAME_B) == "value A\n")
    check("store still has exactly one live file", len(ses.dev.names()) == 1)
    guard.put_checked(alias, "value B", store_call=sc)
    check("alias put succeeds", store_get(ses, T, alias) == "value B\n")
    check("A still intact after alias put", store_get(ses, T, NAME_A) == "value A\n")
    # safe short names
    kc = guard.key("city")
    kd = guard.key("dog_name")
    check("short names have distinct keys", kc != kd)
    cc = guard.check("city")
    cd = guard.check("dog_name")
    check("short names: no collisions", not cc["collision"] and not cd["collision"])
    guard.put_checked("city", "Shanghai, China", store_call=sc)
    guard.put_checked("dog_name", "Biscuit", store_call=sc)
    check("city get", store_get(ses, T, "city") == "Shanghai, China\n")
    check("dog_name get", store_get(ses, T, "dog_name") == "Biscuit\n")
    # same name rewrite is allowed (replace semantics)
    guard.put_checked("city", "New York", store_call=sc)
    check("same-name re-put allowed", store_get(ses, T, "city") == "New York\n")
    # cross-check key formula against the real store device state:
    live_keys = set(ses.dev.names())
    for name in (NAME_A, alias, "city", "dog_name"):
        check(f"guard key matches store key for {name!r}", guard.key(name) in live_keys)
    return ses

# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    run_dir = HERE / "test_runs"
    run_dir.mkdir(exist_ok=True)
    tmpdir = run_dir / time.strftime("%Y%m%d-%H%M%S")
    tmpdir.mkdir()
    import sys as _sys
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"memory_disk test run — {ts}")
    log(f"python {_sys.version.split()[0]} | numpy {np.__version__} | "
        f"memory_disk {MD.__version__}")
    log(f"riscv reference on PYTHONPATH: {RISCV.exists()}")
    body = test_roundtrip(tmpdir)
    test_fingerprint_mismatch(tmpdir, body)
    test_journal_portability()
    test_nameguard()
    dur = time.perf_counter() - t_start
    log("")
    log(f"summary: {len(FAILURES)} failure(s) in {dur:.2f}s")
    if FAILURES:
        log("failed: " + "; ".join(FAILURES))
    log_path = HERE / "test_log.txt"
    log_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    print(f"\nlog written: {log_path}")
    return 1 if FAILURES else 0

if __name__ == "__main__":
    sys.exit(main())
