# -*- coding: utf-8 -*-
"""Memory-disk format v0: fingerprint-bound store snapshots + logical-content journal.

Pure Python + numpy. No model loading (a snapshot body is a .npz whose layout is
exactly the one ``chipformer.rv.chat.save_store_file`` writes, so
``chipformer.rv.chat.load_store_file`` / ``RvChat.load_store`` can read what we
save, and we can read what they save). The header is a sidecar JSON file named
``<path>.json`` next to the body ``<path>``.

Main pieces:

- :func:`fingerprint`: a model/store binding fingerprint (sha256 of config.json +
  chipformer.json + tokenizer.json, streamed, plus the store config and format
  version). A snapshot is only loadable when its fingerprint matches the current
  model+store; otherwise the *logical content* (names + texts) must be migrated by
  journal replay (see :func:`export_journal`, :func:`replay_into_reference_store`).
- :func:`save` / :func:`load` / :func:`verify`: write/read/check a snapshot file.
  load raises :class:`FingerprintMismatch` when the fingerprint differs (unless
  ``force=True``).
- :class:`NameGuard`: API-level guard against the store's silent-overwrite bug
  (key = first NMAX=6 token ids + capped length). It detects when two *distinct*
  names map to the same store key and refuses the put with a suggested alias.

All heavy imports (tokenizers, chipformer.rv.*) are lazy: importing this module
only loads stdlib + numpy.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

__version__ = "0.1.0"
TOOL_VERSION = f"memory_disk.py {__version__}"

MAGIC = "QPUMEM01"
FORMAT_VERSION = 1
HEADER_SUFFIX = ".json"

# --- store layout constants (chipformer.rv.store) mirrored for key computation --------
NMAX = 6          # name tokens kept and compared
LEN_BITS = 3      # name length bits -> max key length value = 7
LEN_MAX = (1 << LEN_BITS) - 1   # 7
ID_BITS = 18      # Qwen3.5 vocab ids fit 18 bits
SPECIAL = 248044  # <|endoftext|> and up: never stored (stored as 0)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_PATH = ROOT / ".data" / "Qwen3.5-4B-Qwenputer-Filesystem" / "tokenizer.json"
DEFAULT_MACHINE_ELF = (ROOT / "repro" / "qwenputer" / "experimental" / "riscv"
                       / "tests" / "data" / "rv" / "calc-bridge.elf")


class MemoryDiskError(Exception):
    """Base error for the memory-disk format."""


class FingerprintMismatch(MemoryDiskError):
    """The snapshot's model/store fingerprint differs from the expected one."""


class NameConflict(MemoryDiskError):
    """A name would silently overwrite a different name sharing the same store key."""


class HeaderError(MemoryDiskError):
    """The sidecar header is missing, malformed, or fails magic/version checks."""


# --------------------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(model_dir, store_config: Optional[dict] = None,
                format_version: int = FORMAT_VERSION) -> dict:
    """The binding fingerprint of a model directory + store config.

    Hashes (streamed) ``config.json``, ``chipformer.json`` and ``tokenizer.json``;
    then folds in the store config dict (canonical JSON) and the format version.
    ``store_config`` defaults to ``chipformer.json["store"]`` (or {} when absent).
    Returns a dict whose ``fp`` key is the overall sha256 hex digest.
    """
    d = Path(model_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"model dir not found: {d}")
    files = {}
    for name in ("config.json", "chipformer.json", "tokenizer.json"):
        p = d / name
        if not p.is_file():
            raise FileNotFoundError(f"missing model file for fingerprint: {p}")
        files[name] = _file_sha256(p)
    if store_config is None:
        try:
            store_config = json.loads((d / "chipformer.json").read_text(encoding="utf-8")).get("store") or {}
        except Exception:
            store_config = {}
    h = hashlib.sha256()
    for name in ("config.json", "chipformer.json", "tokenizer.json"):
        h.update(name.encode("utf-8"))
        h.update(files[name].encode("ascii"))
    h.update(b"store_config")
    h.update(_canonical_json(store_config).encode("utf-8"))
    h.update(b"format_version")
    h.update(str(format_version).encode("ascii"))
    return {
        "format_version": format_version,
        "files": files,
        "store_config": store_config,
        "fp": h.hexdigest(),
    }


# --------------------------------------------------------------------------------------
# snapshot body (npz) parsing / writing — layout-compatible with chat.save_store_file
# --------------------------------------------------------------------------------------

def parse_snapshot_npz(path) -> dict:
    """Read a raw .npz snapshot (no header) into ``{head: [{row: ndarray}]}``.

    Same result as ``chipformer.rv.chat.load_store_file`` (keys ``<head>/n`` for
    the value-head count, ``<head>/<x>/<r>`` for each nonzero state row).
    """
    with np.load(path) as z:
        files = list(z.files)
        snap: dict = {}
        for k in files:
            if k.endswith("/n"):
                snap[k[:-2]] = [{} for _ in range(int(z[k]))]
        for k in files:
            if not k.endswith("/n"):
                head, x, r = k.rsplit("/", 2)
                snap[head][int(x)][int(r)] = z[k]
    return snap


def _snapshot_array_items(snapshot: dict) -> list[tuple[str, Any]]:
    """The npz arrays for a snapshot, in the deterministic order save_store_file uses."""
    items: list[tuple[str, Any]] = []
    for name, vs in snapshot.items():
        if not isinstance(vs, list):
            raise TypeError(f"snapshot value for {name!r} must be a list of per-head dicts")
        items.append((f"{name}/n", np.array(len(vs))))
        for x, rows in enumerate(vs):
            for r in sorted(int(r) for r in rows.keys()):
                arr = np.asarray(rows[r], dtype=np.float64)
                if arr.shape != (128,):
                    raise ValueError(f"{name}/{x}/{r}: expected a 128-d row, got {arr.shape}")
                items.append((f"{name}/{x}/{r}", arr))
    return items


def _build_body_npz_bytes(snapshot: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **dict(_snapshot_array_items(snapshot)))
    return buf.getvalue()


def _write_atomic(final: Path, data: bytes) -> None:
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, final)


# --------------------------------------------------------------------------------------
# save / load / verify
# --------------------------------------------------------------------------------------

def save(snapshot: dict, path, fp: dict, journal: Optional[list] = None) -> Path:
    """Write a snapshot: body .npz + sidecar header .json (magic QPUMEM01).

    ``fp`` is the dict returned by :func:`fingerprint`. ``journal`` (optional)
    is a list of ``{"name": ..., "text": ...}`` items that lets a later load
    migrate the logical content when the fingerprint no longer matches.
    Returns the body path.
    """
    body_path = Path(path)
    body_bytes = _build_body_npz_bytes(snapshot)
    header = {
        "magic": MAGIC,
        "format_version": FORMAT_VERSION,
        "tool_version": TOOL_VERSION,
        "fingerprint": fp,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "body_bytes": len(body_bytes),
        "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "journal": export_journal(journal) if journal is not None else [],
    }
    _write_atomic(body_path, body_bytes)
    _write_atomic(Path(str(body_path) + HEADER_SUFFIX),
                  json.dumps(header, indent=2, ensure_ascii=False).encode("utf-8"))
    return body_path


def _read_header(body_path: Path) -> dict:
    header_path = Path(str(body_path) + HEADER_SUFFIX)
    if not header_path.is_file():
        raise HeaderError(f"missing sidecar header: {header_path}")
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HeaderError(f"header is not valid JSON: {header_path}: {exc}") from exc
    if not isinstance(header, dict) or header.get("magic") != MAGIC:
        raise HeaderError(f"bad magic (expected {MAGIC!r}): {header_path}")
    v = header.get("format_version")
    if v != FORMAT_VERSION:
        raise HeaderError(f"unsupported format_version {v!r} (this tool supports {FORMAT_VERSION})")
    return header


def load(path, expected_fp: Optional[dict] = None, force: bool = False) -> dict:
    """Verify and return the snapshot (compatible with chat.load_store_file).

    Raises :class:`FingerprintMismatch` when ``expected_fp`` is given and its
    ``fp`` differs from the header's (unless ``force=True``).
    """
    body_path = Path(path)
    header = _read_header(body_path)
    body_digest = hashlib.sha256(body_path.read_bytes()).hexdigest()
    if body_digest != header.get("body_sha256"):
        raise MemoryDiskError(f"snapshot body checksum mismatch: {body_path}")
    if expected_fp is not None:
        want = (expected_fp or {}).get("fp")
        got = (header.get("fingerprint") or {}).get("fp")
        if want is not None and got is not None and want != got and not force:
            raise FingerprintMismatch(
                f"fingerprint mismatch: expected {want[:16]}..., header has {got[:16]}... "
                f"({body_path}); load with force=True only for debugging, or migrate by "
                f"journal replay")
    return parse_snapshot_npz(body_path)


def verify(path, expected_fp: Optional[dict] = None) -> dict:
    """Inspect a snapshot file: header + booleans (never raises for bad data)."""
    body_path = Path(path)
    out = {"path": str(body_path),
           "header": None, "magic_ok": False, "version_ok": False,
           "checksum_ok": False, "fingerprint_match": None, "error": None}
    try:
        header = _read_header(body_path)
    except HeaderError as exc:
        out["error"] = str(exc)
        return out
    out["header"] = header
    out["magic_ok"] = True
    out["version_ok"] = True
    try:
        digest = hashlib.sha256(body_path.read_bytes()).hexdigest()
        out["checksum_ok"] = digest == header.get("body_sha256")
    except OSError as exc:
        out["error"] = f"cannot read body: {exc}"
        return out
    if expected_fp is not None:
        want = (expected_fp or {}).get("fp")
        got = (header.get("fingerprint") or {}).get("fp")
        if want is not None and got is not None:
            out["fingerprint_match"] = want == got
    return out


# --------------------------------------------------------------------------------------
# journal (logical content)
# --------------------------------------------------------------------------------------

def export_journal(names_texts) -> list[dict]:
    """Normalize a list of ``{name, text}`` (or ``(name, text)``) into a journal.

    The order is the caller's order; replaying a journal into a fresh store
    re-creates the logical content (names + texts) under any seed/format.
    """
    out = []
    for item in names_texts:
        if isinstance(item, dict):
            if "name" not in item or "text" not in item:
                raise ValueError(f"journal item must have \"name\" and \"text\": {item!r}")
            name, text = item["name"], item["text"]
        else:
            name, text = item
        if not isinstance(name, str) or not isinstance(text, str):
            raise TypeError(f"journal names/texts must be str: {item!r}")
        out.append({"name": name, "text": text})
    return out


def _reference_components(tokenizer_path, machine_elf):
    """Build the reference (chipformer.rv.store) pieces: Text, machine, Config."""
    from chipformer.rv import bridge as BR
    from chipformer.rv import elf, ref
    from chipformer.rv import store as ST

    T = BR.Text(tokenizer_path)
    m = ref.Machine.from_program(elf.program(str(machine_elf)))
    BR.park(m)
    return T, m, ST


def replay_into_reference_store(names_texts, *, seed: int = 2035,
                                store_config: Optional[dict] = None,
                                tokenizer_path=None, machine_elf=None,
                                text_encode: Optional[Callable[[str], list[int]]] = None):
    """Replay a journal into a fresh reference store session (CPU-only).

    Returns ``(session, results)``. ``session`` is a
    ``chipformer.rv.store.StoreSession`` whose ``dev`` now holds the files;
    ``results`` is one dict per put: ``{name, answer, op, found, full}``.
    Replaying the same journal with a different ``seed`` yields the same logical
    content (get() returns the same texts) even though the raw state differs.
    """
    journal = export_journal(names_texts)
    T, m, ST = _reference_components(tokenizer_path or DEFAULT_TOKENIZER_PATH,
                                      machine_elf or DEFAULT_MACHINE_ELF)
    cfg = ST.Config(seed=seed, **(store_config or {}))
    ses = ST.StoreSession(T, m, cfg)
    results = []
    for item in journal:
        name_toks = text_encode(item["name"]) if text_encode else T.encode(item["name"])
        text_toks = text_encode(item["text"]) if text_encode else T.encode(item["text"])
        start = len(ses.toks)
        ans = ses.call("put", name_toks, text_toks)
        printed = ses.printed(start)
        results.append({"name": item["name"], "answer": printed[0] if printed else None,
                        "op": ans[0], "found": ans[1], "full": ans[2]})
    return ses, results


def reference_get(ses, name, *, text_encode: Optional[Callable[[str], list[int]]] = None,
                  tokenizer_path=None):
    """get(name) on a reference session; returns the printed body (text or "not found\n")."""
    T = getattr(ses, "_T", None)
    if T is None:
        T, _m, _ST = _reference_components(tokenizer_path or DEFAULT_TOKENIZER_PATH,
                                            DEFAULT_MACHINE_ELF)
        ses._T = T
    name_toks = text_encode(name) if text_encode else T.encode(name)
    start = len(ses.toks)
    ses.call("get", name_toks)
    printed = ses.printed(start)
    return printed[0] if printed else None


# --------------------------------------------------------------------------------------
# NameGuard
# --------------------------------------------------------------------------------------

def stored(qid: int) -> int:
    """The stored token id (specials become 0), as the store stores them."""
    return qid if qid < SPECIAL else 0


class TokenizerKey:
    """Name-key computation over a tokenizer.json (the store's captured key rule).

    A put call spells a name as ``<parameter=name>\n<name>\n</parameter>``; the
    store captures the name tokens, the trailing newline, and (for short names)
    the closer's ``</`` token, keeping the first NMAX ids plus a capped length:
    key = (min(len(tokens)+1, 7), first 6 stored ids of tokens + [NL, "</"]).
    """

    def __init__(self, tokenizer_path=None):
        from tokenizers import Tokenizer

        self.path = Path(tokenizer_path or DEFAULT_TOKENIZER_PATH)
        self.tok = Tokenizer.from_file(str(self.path))
        self._nl = self.tok.encode("\n", add_special_tokens=False).ids[0]
        self._lt_close = self.tok.encode("</", add_special_tokens=False).ids[0]

    def key(self, name: str):
        toks = self.tok.encode(name, add_special_tokens=False).ids
        n = min(len(toks) + 1, LEN_MAX)
        captured = toks + [self._nl, self._lt_close]
        t6 = tuple(stored(q) for q in captured[:NMAX])
        t6 += (0,) * (NMAX - len(t6))
        return n, t6


class NameGuard:
    """Detects names whose store keys collide (the silent-overwrite bug).

    Usage:

        guard = NameGuard(tokenizer_path)
        guard.register("city")
        guard.check("dog_name")      # -> {collision: False, ...}
        guard.put_checked("city", "x", store_call=put_fn)   # raises NameConflict

    ``store_call`` (optional) is ``f(name, text)`` that performs the real put
    after the guard passes.
    """

    def __init__(self, tokenizer_path=None, *, key_cls=TokenizerKey):
        self.keyer = key_cls(tokenizer_path)
        self._by_key: dict = {}      # key -> canonical name
        self._names: dict = {}       # name -> key

    # -- key helpers -----------------------------------------------------------
    def key(self, name: str):
        return self.keyer.key(name)

    # -- registry --------------------------------------------------------------
    def register(self, name: str) -> tuple:
        """Record a name as present in the store (e.g. after an external successful put)."""
        key = self.key(name)
        self._by_key.setdefault(key, name)
        self._names[name] = key
        return key

    def known_names(self) -> list[str]:
        return list(self._names)

    # -- checks ----------------------------------------------------------------
    def check(self, name: str) -> dict:
        key = self.key(name)
        existing = self._by_key.get(key)
        return {
            "name": name,
            "key": key,
            "tokens": self.keyer.tok.encode(name, add_special_tokens=False).ids,
            "collision": existing is not None and existing != name,
            "existing_name": existing,
            "suggestion": self.suggest(name) if existing is not None and existing != name else None,
        }

    def suggest(self, name: str) -> str:
        """A deterministic alias whose key is not already taken.

        Appending a suffix changes the key only for short names (length is
        capped at 7); for names of >= 6 tokens we must change the head, so the
        second family prepends a bracket tag.
        """
        for i in range(1, 9):
            cand = f"{name} ({i})"
            if self.key(cand) not in self._by_key:
                return cand
        for i in range(1, 9):
            cand = f"[{i}] {name}"
            if self.key(cand) not in self._by_key:
                return cand
        raise MemoryDiskError(f"no free alias for {name!r}")

    def put_checked(self, name: str, text: str,
                    store_call: Optional[Callable[[str, str], Any]] = None):
        """put() that refuses to overwrite a *different* name sharing the key.

        Raises :class:`NameConflict` with a suggested unique alias; otherwise
        runs ``store_call(name, text)`` (if given) and registers the name.
        """
        key = self.key(name)
        existing = self._by_key.get(key)
        if existing is not None and existing != name:
            raise NameConflict(
                f"name {name!r} would silently overwrite {existing!r}: both map to store "
                f"key {key} (first {NMAX} tokens + length). Suggested unique alias: "
                f"{self.suggest(name)!r}")
        result = store_call(name, text) if store_call is not None else None
        self._by_key.setdefault(key, name)
        self._names[name] = key
        return result


if __name__ == "__main__":
    import sys
    print(f"memory_disk.py {__version__}: magic={MAGIC} format_version={FORMAT_VERSION}")
    print("usage: import memory_disk (see module docstring)", file=sys.stderr)
