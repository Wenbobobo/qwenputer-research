# -*- coding: utf-8 -*-
"""Comprehensive validation suite for analysis/memory_disk/safe_store.py (S5.3).

Runs CPU-only on reference store (no model loading).

Test Groups:
  1. D Collision Example:
     - 'the locker code for the safe room in' vs 'the locker code for the safe in room'
     - Verified: both stored independently, get returns distinct texts, list preserves both.
  2. Normal Short Names:
     - Verified: 'city', 'dog_name', 'pin', etc. remain UNCHANGED (canonical == logical).
     - Update / replace semantics verified.
  3. Delete Semantics:
     - Deleting one colliding entry preserves the other.
     - Deleting non-existent returns not found cleanly.
  4. 512 Adversarial Long Names:
     - 512 distinct long names sharing the EXACT same first 6 tokens.
     - Raw store: 1 unique key, 511 overwritten.
     - SafeStore: 512 unique keys, 512 live files, 512 gets matching 100%.
  5. Persistence & Snapshot Roundtrip:
     - Save snapshot + sidecar mapping table.
     - Load in fresh SafeStore: all logical names and texts intact.
  6. Collision Probability & Corpus Statistics (H2 corpus):
     - Measure raw vs 1-token (18-bit) vs 2-token (36-bit) vs hex tag.
     - Verified theoretical and empirical collision numbers.

Consolidated log written to analysis/memory_disk/test_safe_store_log.txt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "repro" / "qwenputer" / "experimental" / "riscv"))
sys.path.insert(0, str(ROOT / "analysis" / "experiments"))

import memory_disk as MD  # noqa: E402
import safe_store as SS  # noqa: E402
import h2_names  # noqa: E402

NAME_A = "the locker code for the safe room in"
NAME_B = "the locker code for the safe in room"
TEXT_A = "locker code is 4471-AZ"
TEXT_B = "locker code is 4472-AZ"

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
# 1. D Collision Example
# --------------------------------------------------------------------------------------

def test_d_collision_example():
    log("\n=== 1. D Collision Example (Independent Storage of Colliding Long Names) ===")
    safe = SS.SafeStore(seed=2035)

    raw_k_a = safe.raw_key(NAME_A)
    raw_k_b = safe.raw_key(NAME_B)
    log(f"  raw_key(NAME_A) = {raw_k_a}")
    log(f"  raw_key(NAME_B) = {raw_k_b}")
    check("D example raw keys are identical (the bug exists)", raw_k_a == raw_k_b)

    # Put A
    res_a = safe.put(NAME_A, TEXT_A)
    check("Put A succeeds", res_a["answer"] == "ok\n" and not res_a["full"])
    check("A is not disambiguated (first entry)", not res_a["disambiguated"])
    check("A canonical name equals logical name", res_a["canonical_name"] == NAME_A)

    # Put B
    res_b = safe.put(NAME_B, TEXT_B)
    check("Put B succeeds", res_b["answer"] == "ok\n" and not res_b["full"])
    check("B is disambiguated (collision detected)", res_b["disambiguated"])
    check("B canonical name has tag prefix", res_b["canonical_name"].startswith("#") and res_b["canonical_name"] != NAME_B)
    check("B canonical store key differs from A", safe.store_key_of(NAME_B) != safe.store_key_of(NAME_A))
    log(f"  A canonical: {res_a['canonical_name']!r} -> key: {safe.store_key_of(NAME_A)}")
    log(f"  B canonical: {res_b['canonical_name']!r} -> key: {safe.store_key_of(NAME_B)}")

    # Verify both are alive in store device
    live_keys = safe.used_store_keys()
    check("Underlying store has 2 distinct active files", len(live_keys) == 2)
    check("Store contains A's key", safe.store_key_of(NAME_A) in live_keys)
    check("Store contains B's key", safe.store_key_of(NAME_B) in live_keys)

    # Get A and Get B
    get_a = safe.get(NAME_A)
    get_b = safe.get(NAME_B)
    log(f"  get(NAME_A) -> {get_a!r}")
    log(f"  get(NAME_B) -> {get_b!r}")
    check("get(NAME_A) returns TEXT_A exactly (not overwritten!)", get_a == TEXT_A + "\n")
    check("get(NAME_B) returns TEXT_B exactly", get_b == TEXT_B + "\n")

    # List
    listed = safe.list()
    log(f"  list() -> {listed}")
    check("list() contains NAME_A", NAME_A in listed)
    check("list() contains NAME_B", NAME_B in listed)
    check("list() returns untruncated logical names", all(nm in (NAME_A, NAME_B) for nm in listed))


# --------------------------------------------------------------------------------------
# 2. Normal Short Names
# --------------------------------------------------------------------------------------

def test_normal_short_names():
    log("\n=== 2. Normal Short Names (Unchanged & Replace Semantics) ===")
    safe = SS.SafeStore(seed=2035)

    short_items = [
        ("city", "Shanghai, China"),
        ("dog_name", "Biscuit"),
        ("pin", "2048"),
        ("color", "blue"),
    ]

    for name, text in short_items:
        res = safe.put(name, text)
        check(f"Put {name} succeeds", res["answer"] == "ok\n")
        check(f"{name} canonical == logical (unchanged)", res["canonical_name"] == name and not res["disambiguated"])
        check(f"{name} store key matches raw key", safe.store_key_of(name) == safe.raw_key(name))

    for name, text in short_items:
        got = safe.get(name)
        check(f"Get {name} returns original text", got == text + "\n")

    # Replace / update semantics
    res_up = safe.put("city", "New York, USA")
    check("Update 'city' succeeds", res_up["answer"] == "ok\n")
    check("'city' canonical name preserved on update", res_up["canonical_name"] == "city")
    check("Get 'city' returns updated text", safe.get("city") == "New York, USA\n")
    check("File count remains 4 after update", len(safe.list()) == 4)


# --------------------------------------------------------------------------------------
# 3. Delete Semantics
# --------------------------------------------------------------------------------------

def test_delete_semantics():
    log("\n=== 3. Delete Semantics ===")
    safe = SS.SafeStore(seed=2035)

    safe.put(NAME_A, TEXT_A)
    safe.put(NAME_B, TEXT_B)
    safe.put("city", "Shanghai")

    check("Initial list has 3 files", len(safe.list()) == 3)

    # Delete A (colliding name)
    del_a = safe.delete(NAME_A)
    check("Delete A succeeds", del_a["found"] is True)
    check("Get A after delete returns None", safe.get(NAME_A) is None)
    check("Get B after deleting A is STILL accessible (isolated!)", safe.get(NAME_B) == TEXT_B + "\n")
    check("Get city still accessible", safe.get("city") == "Shanghai\n")

    listed_after = safe.list()
    check("List after delete does not contain A", NAME_A not in listed_after)
    check("List after delete contains B and city", NAME_B in listed_after and "city" in listed_after)

    # Delete non-existent
    del_none = safe.delete("non_existent_key")
    check("Delete non-existent returns found=False", del_none["found"] is False)


# --------------------------------------------------------------------------------------
# 4. 512 Adversarial Long Names
# --------------------------------------------------------------------------------------

def test_512_adversarial_names():
    log("\n=== 4. 512 Adversarial Long Names (Stress & Zero-Collision Proof) ===")
    safe = SS.SafeStore(seed=2035)

    prefix = "the locker code for the safe room in the underground bunker"
    N = 512

    names = [f"{prefix} file_index_{i:04d} extra_tokens" for i in range(N)]
    texts = [f"secret_content_payload_{i:04d}" for i in range(N)]

    # Confirm that raw store would completely fail:
    raw_keys = [safe.raw_key(n) for n in names]
    raw_unique = len(set(raw_keys))
    log(f"  Raw store unique keys across {N} adversarial names: {raw_unique} (would overwrite {N - raw_unique} files)")
    check("Raw store has exactly 1 key for all 512 names", raw_unique == 1)

    t0 = time.perf_counter()
    put_failures = 0
    disambiguated = 0
    for i in range(N):
        res = safe.put(names[i], texts[i])
        if res["full"] or res["answer"] != "ok\n":
            put_failures += 1
        if res["disambiguated"]:
            disambiguated += 1

    dur_put = time.perf_counter() - t0
    log(f"  Put {N} adversarial names in {dur_put:.3f}s ({dur_put/N*1000:.2f} ms/put)")
    log(f"  Disambiguated count: {disambiguated} / {N} (1st entry kept raw, {N-1} tagged)")
    check("All 512 puts succeed", put_failures == 0)
    check("Exactly 511 names disambiguated", disambiguated == N - 1)

    # Check store capacity
    live_keys = safe.used_store_keys()
    log(f"  Live keys in store device: {len(live_keys)} / {N}")
    check("Store contains EXACTLY 512 live files (100% capacity)", len(live_keys) == N)

    # Verify gets
    t1 = time.perf_counter()
    get_mismatches = 0
    for i in range(N):
        got = safe.get(names[i])
        if got != texts[i] + "\n":
            get_mismatches += 1

    dur_get = time.perf_counter() - t1
    log(f"  Verified {N} gets in {dur_get:.3f}s ({dur_get/N*1000:.2f} ms/get), mismatches: {get_mismatches}")
    check("All 512 files retrieved with 100% exact text fidelity", get_mismatches == 0)

    # Verify list
    listed = safe.list()
    check("List returns exactly 512 items", len(listed) == N)
    check("List preserves all logical names", set(listed) == set(names))


# --------------------------------------------------------------------------------------
# 5. Persistence & Snapshot Roundtrip
# --------------------------------------------------------------------------------------

def test_persistence_roundtrip(tmpdir: Path):
    log("\n=== 5. Persistence & Mapping Table Roundtrip ===")
    safe1 = SS.SafeStore(seed=2035)

    safe1.put(NAME_A, TEXT_A)
    safe1.put(NAME_B, TEXT_B)
    safe1.put("city", "Shanghai")
    safe1.put("dog", "Biscuit")

    snap_file = tmpdir / "safe_snap.pkl"
    safe1.save_device_snapshot(snap_file)

    map_file = tmpdir / "safe_snap.pkl.map.json"
    json_file = tmpdir / "safe_snap.pkl.json"

    check("Snapshot body file exists", snap_file.is_file())
    check("Mapping sidecar file exists", map_file.is_file())
    check("Header sidecar file exists", json_file.is_file())

    map_data = json.loads(map_file.read_text(encoding="utf-8"))
    log(f"  Mapping sidecar entries: {len(map_data['entries'])}")
    check("Mapping sidecar has 4 entries", len(map_data["entries"]) == 4)

    # Load into fresh SafeStore
    safe2 = SS.SafeStore.load_device_snapshot(snap_file, seed=2035)

    check("Restored safe2 get(NAME_A) matches", safe2.get(NAME_A) == TEXT_A + "\n")
    check("Restored safe2 get(NAME_B) matches", safe2.get(NAME_B) == TEXT_B + "\n")
    check("Restored safe2 get('city') matches", safe2.get("city") == "Shanghai\n")
    check("Restored safe2 get('dog') matches", safe2.get("dog") == "Biscuit\n")
    check("Restored safe2 list() matches", set(safe2.list()) == {NAME_A, NAME_B, "city", "dog"})


# --------------------------------------------------------------------------------------
# 6. Collision Probability & Corpus Statistics (H2 Corpus)
# --------------------------------------------------------------------------------------

def test_collision_stats_corpus():
    log("\n=== 6. Collision Statistics & Probability Analysis (H2 Corpus) ===")
    corpus = h2_names.corpus
    adv_subset = [nm for nm in corpus if h2_names.cat[nm] == "adversarial_prefix"]
    adv_512 = [f"the locker code for the safe room in item_{i:04d}" for i in range(512)]

    log("\n-- A. Full H2 Corpus (2000 names) --")
    res_2000 = SS.measure_tagging_schemes(corpus)
    log(json.dumps(res_2000, indent=2))
    check("H2 2000 raw store has collisions", res_2000["raw_store"]["collisions"] > 0)
    check("H2 2000 2-token tag has 0 collisions", res_2000["tag_2token_36bit"]["measured_collisions_avg"] == 0.0)
    check("H2 2000 SafeStore protocol has 0 collisions", res_2000["safestore_protocol"]["collisions"] == 0)

    log("\n-- B. H2 Adversarial Subset (400 names sharing 40 long prefixes) --")
    res_adv400 = SS.measure_tagging_schemes(adv_subset)
    log(json.dumps(res_adv400, indent=2))
    check("H2 adv400 raw store collision rate is high (0.90)", res_adv400["raw_store"]["collision_rate"] == 0.9)
    check("H2 adv400 SafeStore protocol has 0 collisions", res_adv400["safestore_protocol"]["collisions"] == 0)

    log("\n-- C. Adversarial 512 (512 names sharing identical prefix) --")
    res_adv512 = SS.measure_tagging_schemes(adv_512)
    log(json.dumps(res_adv512, indent=2))
    check("Adv 512 raw store has 511 collisions (99.8%)", res_adv512["raw_store"]["collisions"] == 511)
    check("Adv 512 2-token (36-bit) tag measured 0 collisions", res_adv512["tag_2token_36bit"]["measured_collisions_avg"] == 0.0)
    check("Adv 512 SafeStore protocol has 0 collisions", res_adv512["safestore_protocol"]["collisions"] == 0)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    run_dir = HERE / "test_runs"
    run_dir.mkdir(exist_ok=True)
    tmpdir = run_dir / time.strftime("%Y%m%d-%H%M%S")
    tmpdir.mkdir()

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"SafeStore test run — {ts}")
    log(f"Python {sys.version.split()[0]} | SafeStore S5.3 KeyFix Validation")
    log(f"Reference Store: chipformer.rv.store (CPU-only, no HF model)")

    test_d_collision_example()
    test_normal_short_names()
    test_delete_semantics()
    test_512_adversarial_names()
    test_persistence_roundtrip(tmpdir)
    test_collision_stats_corpus()

    dur = time.perf_counter() - t_start
    log("\n" + "=" * 60)
    log(f"SUMMARY: {len(FAILURES)} failure(s) in {dur:.2f}s")
    if FAILURES:
        log("FAILED: " + "; ".join(FAILURES))
    else:
        log("ALL TESTS PASSED!")

    log_path = HERE / "test_safe_store_log.txt"
    log_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    print(f"\nLog written: {log_path}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
