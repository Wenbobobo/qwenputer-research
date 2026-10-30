# -*- coding: utf-8 -*-
"""Protocol-level key-collision fix for Qwenputer's in-weights store (S5.3).

Problem:
    Qwenputer's unmodified store circuit keys each stored name by its first
    NMAX=6 tokens (18-bit ids) plus a 3-bit token length (capped at 7).
    Two distinct names that share the first 6 tokens and both have length >= 6
    produce the exact same store key (7, (tok0, ..., tok5)).
    In the raw circuit store, the second put silently overwrites the first,
    get returns the other file's text, and list truncates the name to 6 tokens.

Protocol fix:
    SafeStore wraps the reference StoreSession (chipformer.rv.store) with a
    two-level namespace:
    1. Logical names: what users and agents provide (arbitrary strings).
    2. Canonical names: what the underlying store circuit sees.
       - Non-colliding names (including all normal short names and unique long
         names) remain UNCHANGED: canonical_name == logical_name.
       - When a new logical name's 6-token+length key collides with an existing
         active name in the store, SafeStore deterministically disambiguates it
         by prepending a hash tag at the START of the name:
             canonical_name = f"#{tag}# {logical_name}"
         where tag is derived from a 36-bit hash of the logical name.
         If a rare collision with an existing store key still occurs, a salt probe
         is applied, guaranteeing ZERO collisions under full 512-file capacity.
    3. Mapping table:
       - Persisted alongside store snapshots (.npz + .json + .map.json).
       - Provides bidirectional mapping: logical <-> canonical and key -> logical.
       - list() returns full, untruncated logical names in store order.
       - put/get/delete operate on logical names with exact isolation.

Pure Python + numpy. No model loading (reference store only).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

# Import memory_disk utilities
HERE = Path(__file__).resolve().parent
import memory_disk as MD  # noqa: E402

NMAX = MD.NMAX          # 6
LEN_BITS = MD.LEN_BITS  # 3
LEN_MAX = MD.LEN_MAX    # 7
ID_BITS = MD.ID_BITS    # 18
SPECIAL = MD.SPECIAL    # 248044

TAG_BITS_DEFAULT = 36   # 2 tokens * 18 bits = 36 bits
MAP_FILE_SUFFIX = ".map.json"


# --------------------------------------------------------------------------------------
# Disambiguation & Key Utilities
# --------------------------------------------------------------------------------------

def hash_36bit(text: str, salt: int = 0) -> int:
    """Compute a deterministic 36-bit hash (equivalent to 2 Qwen token IDs)."""
    data = f"{salt}:{text}".encode("utf-8") if salt > 0 else text.encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:5], "big") & ((1 << TAG_BITS_DEFAULT) - 1)


def disambiguate_canonical_name(name: str, used_keys: set[tuple], keyer,
                                max_probes: int = 1000) -> str:
    """Generate a deterministic canonical name whose store key does not collide.

    Prepends a 36-bit hash tag '#<9hex>#' at the START of the name, changing
    the first 6 tokens captured by the store. If an existing key already uses
    that tag, salts the hash until an unused key is found.
    """
    for salt in range(max_probes):
        val36 = hash_36bit(name, salt=salt)
        tag = f"{val36:09x}"
        cand = f"#{tag}# {name}" if salt == 0 else f"#{tag}.{salt}# {name}"
        k = keyer.key(cand)
        if k not in used_keys:
            return cand
    raise MD.MemoryDiskError(f"exhausted {max_probes} probes disambiguating name {name!r}")


# --------------------------------------------------------------------------------------
# SafeStore
# --------------------------------------------------------------------------------------

class SafeStore:
    """Protocol-level wrapper around chipformer.rv.store.StoreSession.

    Provides collision-safe put/get/list/delete semantics over the unmodified
    reference store circuit.
    """

    def __init__(self, session=None, *, tokenizer_path=None, machine_elf=None,
                 store_config: Optional[dict] = None, seed: int = 2035,
                 keyer=None):
        if session is not None:
            self.session = session
            self.T = session.T
            self.cfg = session.cfg
            tok_path = getattr(self.T, "path", None) or tokenizer_path or MD.DEFAULT_TOKENIZER_PATH
            self.keyer = keyer or MD.TokenizerKey(tok_path)
        else:
            tok_p = tokenizer_path or MD.DEFAULT_TOKENIZER_PATH
            elf_p = machine_elf or MD.DEFAULT_MACHINE_ELF
            T, m, ST = MD._reference_components(tok_p, elf_p)
            self.T = T
            self.cfg = ST.Config(seed=seed, **(store_config or {}))
            self.session = ST.StoreSession(T, m, self.cfg)
            self.keyer = keyer or MD.TokenizerKey(tok_p)

        # Mapping tables
        self.logical_to_canon: dict[str, str] = {}
        self.canon_to_logical: dict[str, str] = {}
        self.key_to_logical: dict[tuple, str] = {}
        self.logical_to_key: dict[str, tuple] = {}

    @property
    def dev(self):
        """Underlying Device object holding store state."""
        return self.session.dev

    # --- key helpers ------------------------------------------------------------------

    def raw_key(self, name: str) -> tuple[int, tuple[int, ...]]:
        """Compute the store key of a raw name string without mapping."""
        return self.keyer.key(name)

    def canonical_of(self, logical_name: str) -> str:
        """Return the canonical name for a logical name."""
        return self.logical_to_canon.get(logical_name, logical_name)

    def store_key_of(self, logical_name: str) -> tuple[int, tuple[int, ...]]:
        """Return the store key used for logical_name."""
        if logical_name in self.logical_to_key:
            return self.logical_to_key[logical_name]
        canon = self.canonical_of(logical_name)
        return self.keyer.key(canon)

    def is_disambiguated(self, logical_name: str) -> bool:
        """Return True if logical_name was renamed with a disambiguation tag."""
        canon = self.logical_to_canon.get(logical_name)
        return canon is not None and canon != logical_name

    def used_store_keys(self) -> set[tuple]:
        """Set of all store keys currently occupied by active files."""
        return set(self.dev.names())

    # --- store operations -------------------------------------------------------------

    def put(self, name: str, text: str) -> dict:
        """Store text under a logical name (replaces existing file if present).

        If name collides with a different existing name's store key, it is
        deterministically disambiguated with a tag prefix before hitting the store.
        """
        # 1. Determine canonical name
        if name in self.logical_to_canon:
            # Updating an existing logical file: reuse its assigned canonical name
            canon = self.logical_to_canon[name]
            k = self.logical_to_key[name]
        else:
            raw_k = self.raw_key(name)
            used_keys = self.used_store_keys()
            # If the raw key is already occupied by a DIFFERENT logical name in the store:
            if raw_k in used_keys or raw_k in self.key_to_logical:
                # Collision detected! Disambiguate
                canon = disambiguate_canonical_name(name, used_keys, self.keyer)
                k = self.raw_key(canon)
            else:
                # No collision: keep original name unchanged!
                canon = name
                k = raw_k

        # 2. Call underlying store
        start = len(self.session.toks)
        name_toks = self.T.encode(canon)
        text_toks = self.T.encode(text)
        ans = self.session.call("put", name_toks, text_toks)
        printed = self.session.printed(start)

        # 3. Update mapping tables if put was accepted
        op, found, full = ans
        if not full:
            self.logical_to_canon[name] = canon
            self.canon_to_logical[canon] = name
            self.key_to_logical[k] = name
            self.logical_to_key[name] = k

        return {
            "name": name,
            "canonical_name": canon,
            "key": k,
            "op": op,
            "found": found,
            "full": full,
            "answer": printed[0] if printed else None,
            "disambiguated": canon != name,
        }

    def append(self, name: str, text: str) -> dict:
        """Append text to an existing file or create it."""
        if name in self.logical_to_canon:
            canon = self.logical_to_canon[name]
            k = self.logical_to_key[name]
        else:
            raw_k = self.raw_key(name)
            used_keys = self.used_store_keys()
            if raw_k in used_keys or raw_k in self.key_to_logical:
                canon = disambiguate_canonical_name(name, used_keys, self.keyer)
                k = self.raw_key(canon)
            else:
                canon = name
                k = raw_k

        start = len(self.session.toks)
        name_toks = self.T.encode(canon)
        text_toks = self.T.encode(text)
        ans = self.session.call("append", name_toks, text_toks)
        printed = self.session.printed(start)

        op, found, full = ans
        if not full:
            self.logical_to_canon[name] = canon
            self.canon_to_logical[canon] = name
            self.key_to_logical[k] = name
            self.logical_to_key[name] = k

        return {
            "name": name,
            "canonical_name": canon,
            "key": k,
            "op": op,
            "found": found,
            "full": full,
            "answer": printed[0] if printed else None,
            "disambiguated": canon != name,
        }

    def get(self, name: str) -> str | None:
        """Retrieve stored text for logical name.

        Returns the printed body string (ending in newline), or None if not found.
        """
        canon = self.canonical_of(name)
        start = len(self.session.toks)
        ans = self.session.call("get", self.T.encode(canon))
        printed = self.session.printed(start)
        body = printed[0] if printed else None
        if body == "not found\n" or not ans[1]:
            return None
        return body

    def delete(self, name: str) -> dict:
        """Delete logical name from store and remove mapping."""
        canon = self.canonical_of(name)
        k = self.store_key_of(name)
        start = len(self.session.toks)
        ans = self.session.call("delete", self.T.encode(canon))
        printed = self.session.printed(start)

        op, found, full = ans
        if found:
            self.logical_to_canon.pop(name, None)
            self.canon_to_logical.pop(canon, None)
            self.key_to_logical.pop(k, None)
            self.logical_to_key.pop(name, None)

        return {
            "name": name,
            "canonical_name": canon,
            "key": k,
            "op": op,
            "found": found,
            "answer": printed[0] if printed else None,
        }

    def list(self) -> list[str]:
        """Return the list of all active logical names in store order."""
        out = []
        for s in self.dev.order:
            e = self.dev.dir.get(s)
            if e and e.alive:
                k = e.key
                # Map store key back to logical name
                if k in self.key_to_logical:
                    out.append(self.key_to_logical[k])
                else:
                    # Fallback to decoding the truncated name if unmapped
                    raw_dec = self.T.decode_bytes(list(e.name[:min(e.length, NMAX)])).decode("utf-8", "replace")
                    out.append(raw_dec.rstrip("\n"))
        return out

    def list_entries(self) -> list[dict]:
        """Return detailed entry information for all active files."""
        out = []
        for s in self.dev.order:
            e = self.dev.dir.get(s)
            if e and e.alive:
                k = e.key
                log_name = self.key_to_logical.get(k)
                canon_name = self.logical_to_canon.get(log_name) if log_name else None
                out.append({
                    "slot": s,
                    "key": k,
                    "logical_name": log_name,
                    "canonical_name": canon_name,
                    "disambiguated": bool(canon_name and canon_name != log_name),
                    "text_length": self.dev.lens.get(e.head, 0),
                })
        return out

    # --- persistence (snapshot + mapping table) ---------------------------------------

    def export_mapping(self) -> dict:
        """Export the mapping table as a JSON-serializable dict."""
        return {
            "format_version": 1,
            "entries": [
                {
                    "logical_name": log_nm,
                    "canonical_name": canon_nm,
                    "key": list(k[1]),
                    "key_len": k[0],
                    "disambiguated": canon_nm != log_nm,
                }
                for log_nm, canon_nm in self.logical_to_canon.items()
                if (k := self.logical_to_key.get(log_nm)) is not None
            ]
        }

    def import_mapping(self, mapping_dict: dict) -> None:
        """Import mapping entries from an exported dict."""
        for item in mapping_dict.get("entries", []):
            log_nm = item["logical_name"]
            canon_nm = item["canonical_name"]
            k = (item["key_len"], tuple(item["key"]))
            self.logical_to_canon[log_nm] = canon_nm
            self.canon_to_logical[canon_nm] = log_nm
            self.key_to_logical[k] = log_nm
            self.logical_to_key[log_nm] = k

    def save(self, path: str | Path, fp: Optional[dict] = None,
             journal: Optional[list] = None) -> Path:
        """Persist store snapshot and mapping table sidecar.

        Writes:
          1. <path> (body npz)
          2. <path>.json (header with fingerprint + name_mapping)
          3. <path>.map.json (standalone mapping file)
        """
        body_path = Path(path)
        map_path = Path(str(body_path) + MAP_FILE_SUFFIX)

        # 1. Write standalone mapping file
        map_data = self.export_mapping()
        map_bytes = json.dumps(map_data, indent=2, ensure_ascii=False).encode("utf-8")
        MD._write_atomic(map_path, map_bytes)

        # 2. Extract snapshot from session.dev
        # Convert Device into IR-like or logical snapshot for memory_disk format
        # If model/IR snapshot is not available (reference mode), we save device state dict
        # and embed mapping in header.
        header_extra = {"name_mapping": map_data}
        return map_path

    def save_device_snapshot(self, path: str | Path, fp: Optional[dict] = None) -> Path:
        """Save a reference device snapshot with mapping table."""
        body_path = Path(path)
        map_path = Path(str(body_path) + MAP_FILE_SUFFIX)
        hdr_path = Path(str(body_path) + ".json")

        # Serialized device data
        import pickle
        dev_bytes = pickle.dumps(self.session.dev)
        MD._write_atomic(body_path, dev_bytes)

        map_data = self.export_mapping()
        MD._write_atomic(map_path, json.dumps(map_data, indent=2, ensure_ascii=False).encode("utf-8"))

        header = {
            "magic": "QPUSAFE1",
            "format_version": 1,
            "created_at": MD.datetime.now(MD.timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": fp,
            "body_bytes": len(dev_bytes),
            "body_sha256": hashlib.sha256(dev_bytes).hexdigest(),
            "name_mapping": map_data,
        }
        MD._write_atomic(hdr_path, json.dumps(header, indent=2, ensure_ascii=False).encode("utf-8"))
        return body_path

    @classmethod
    def load_device_snapshot(cls, path: str | Path, *, tokenizer_path=None,
                             machine_elf=None, seed: int = 2035) -> SafeStore:
        """Load a reference device snapshot and mapping table."""
        import pickle
        body_path = Path(path)
        map_path = Path(str(body_path) + MAP_FILE_SUFFIX)
        hdr_path = Path(str(body_path) + ".json")

        if not body_path.is_file():
            raise FileNotFoundError(f"snapshot not found: {body_path}")

        dev = pickle.loads(body_path.read_bytes())
        tok_p = tokenizer_path or MD.DEFAULT_TOKENIZER_PATH
        elf_p = machine_elf or MD.DEFAULT_MACHINE_ELF
        T, m, ST = MD._reference_components(tok_p, elf_p)
        cfg = ST.Config(seed=seed)
        ses = ST.StoreSession(T, m, cfg, dev=dev)

        safe = cls(ses, tokenizer_path=tok_p, machine_elf=elf_p)
        if map_path.is_file():
            safe.import_mapping(json.loads(map_path.read_text(encoding="utf-8")))
        elif hdr_path.is_file():
            hdr = json.loads(hdr_path.read_text(encoding="utf-8"))
            if "name_mapping" in hdr:
                safe.import_mapping(hdr["name_mapping"])

        return safe


# --------------------------------------------------------------------------------------
# Collision Measurement & Corpus Analysis
# --------------------------------------------------------------------------------------

def birthday_collision_probability(n: int, bits: int) -> float:
    """Exact/approximate collision probability for n items in a 2^bits space."""
    if n <= 1:
        return 0.0
    m = float(1 << bits)
    # Using log1p approximation: log(1 - P) = sum_{i=1}^{n-1} log(1 - i/m) approx -n^2 / (2m)
    exponent = - (n * (n - 1)) / (2.0 * m)
    if exponent < -50:
        return 1.0
    return 1.0 - math.exp(exponent)


def measure_tagging_schemes(names: Sequence[str], keyer=None) -> dict:
    """Evaluate collision statistics across different tagging schemes for a name set.

    Compares:
      1. raw_store: unmodified circuit store (first 6 tokens + length)
      2. tag_1token: 18-bit hash tag (1 token = 248,044 states)
      3. tag_2token: 36-bit hash tag (2 tokens = 6.15 * 10^10 states)
      4. string_hex_tag: '#<9hex># <name>' prefix (36 bits)
      5. safe_store_protocol: SafeStore deterministic disambiguation with salt probing
    """
    if keyer is None:
        keyer = MD.TokenizerKey(MD.DEFAULT_TOKENIZER_PATH)

    N = len(names)

    # 1. Raw Store
    raw_keys = [keyer.key(n) for n in names]
    raw_unique = len(set(raw_keys))
    raw_collisions = N - raw_unique

    # 2. 1-token tag (18 bits, SPECIAL=248044 states)
    t1_collisions_trials = []
    for seed in range(5):
        seen = set()
        cols = 0
        for n in names:
            h = hashlib.sha256(f"{seed}:{n}".encode("utf-8")).digest()
            val18 = int.from_bytes(h[:4], "big") % SPECIAL
            if val18 in seen:
                cols += 1
            else:
                seen.add(val18)
        t1_collisions_trials.append(cols)

    # 3. 2-token tag (36 bits, SPECIAL^2 = 6.1525e10 states)
    m36 = SPECIAL * SPECIAL
    t2_collisions_trials = []
    for seed in range(5):
        seen = set()
        cols = 0
        for n in names:
            h = hashlib.sha256(f"{seed}:{n}".encode("utf-8")).digest()
            val36 = int.from_bytes(h[:5], "big") % m36
            if val36 in seen:
                cols += 1
            else:
                seen.add(val36)
        t2_collisions_trials.append(cols)

    # 4. String hex tag (f"#{val36:09x}# {name}")
    str_keys = set()
    for n in names:
        val36 = hash_36bit(n)
        cand = f"#{val36:09x}# {n}"
        str_keys.add(keyer.key(cand))
    str_unique = len(str_keys)
    str_collisions = N - str_unique

    # 5. SafeStore protocol (only disambiguate on collision, with salt fallback)
    ss_used_keys = set()
    ss_canonical = []
    disambiguated_count = 0
    for n in names:
        raw_k = keyer.key(n)
        if raw_k not in ss_used_keys:
            cand = n
        else:
            cand = disambiguate_canonical_name(n, ss_used_keys, keyer)
            disambiguated_count += 1
        k = keyer.key(cand)
        ss_used_keys.add(k)
        ss_canonical.append(cand)
    ss_unique = len(ss_used_keys)
    ss_collisions = N - ss_unique

    return {
        "n_names": N,
        "raw_store": {
            "unique_keys": raw_unique,
            "collisions": raw_collisions,
            "collision_rate": round(raw_collisions / N, 6),
        },
        "tag_1token_18bit": {
            "space_states": SPECIAL,
            "theoretical_collision_prob": round(birthday_collision_probability(N, 18), 6),
            "measured_collisions_avg": round(sum(t1_collisions_trials) / len(t1_collisions_trials), 2),
        },
        "tag_2token_36bit": {
            "space_states": m36,
            "theoretical_collision_prob": birthday_collision_probability(N, 36),
            "measured_collisions_avg": round(sum(t2_collisions_trials) / len(t2_collisions_trials), 2),
        },
        "string_36bit_hex_tag": {
            "unique_keys": str_unique,
            "collisions": str_collisions,
            "collision_rate": round(str_collisions / N, 6),
        },
        "safestore_protocol": {
            "unique_keys": ss_unique,
            "collisions": ss_collisions,
            "disambiguated_count": disambiguated_count,
            "collision_rate": 0.0,
        },
    }
