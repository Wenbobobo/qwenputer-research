"""Pure-Python copies of the reference protocol in
repro/qwenputer/experimental/riscv/chipformer/rv/store.py (constants, key_bits, Config.slot_of,
StoreScan.name_key, StoreSession.call_tokens, respond() list/get layouts), plus tokenizer helpers.
Copied rather than imported so that the store package (torch/bridge deps) is not needed.
"""
import json
import os

import numpy as np
from tokenizers import Tokenizer

ROOT = "C:/Projects/Qcompute"
MODEL_DIR = os.path.join(ROOT, ".data", "Qwen3.5-4B-Qwenputer-Filesystem")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
RESULTS_PATH = os.path.join(ROOT, "analysis", "experiments", "h2h3_results.json")

# store.py constants
ID_BITS = 18
SPECIAL = 248044
NMAX = 6
LEN_BITS = 3
CMAX = 104
TPR = 14
ROWS_PER_CHUNK = -(-CMAX // TPR)  # 8
Q_NL, Q_EQ, Q_FUNCTION, Q_PARAMETER, Q_LTSLASH, Q_LT, Q_GT = 198, 28, 1628, 15704, 510, 27, 29
Q_TCO, Q_TCC, Q_TRO, Q_TRC, Q_THO, Q_THC = 248058, 248059, 248066, 248067, 248068, 248069
OP_JOINED = {"get": 27362, "list": 39619}
PARAM_JOINED = {"name": 37943, "text": 46696}
TAIL = [Q_TRC, Q_NL, Q_THO, Q_NL, Q_NL, Q_THC, Q_NL, Q_NL]
N_SLOTS_DEFAULT = 1024
N_NAMES_DEFAULT = 512
N_CHUNKS_DEFAULT = 512
SEED = 2035


def load_tokenizer():
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    # sanity: the special ids the store relies on
    assert tok.token_to_id("<tool_call>") == Q_TCO, tok.token_to_id("<tool_call>")
    assert tok.token_to_id("</tool_call>") == Q_TCC
    assert tok.token_to_id("<tool_response>") == Q_TRO
    assert tok.token_to_id("</tool_response>") == Q_TRC
    return tok


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def id_bits(qid):
    if qid is None or qid >= SPECIAL:
        return [0] * ID_BITS
    return [qid >> i & 1 for i in range(ID_BITS)]


def key_bits(key):
    n, toks = key
    out = []
    for q in toks:
        out += id_bits(q)
    return out + [n >> i & 1 for i in range(LEN_BITS)]


class Config:
    """store.py Config (defaults: slot_bits=10, n_chunks=512, n_names=512, seed=2035)."""

    def __init__(self, slot_bits=10, n_chunks=512, n_names=512, hash_bits=None, seed=SEED):
        self.slot_bits, self.n_chunks, self.n_names = slot_bits, n_chunks, n_names
        self.hash_bits, self.seed = hash_bits, seed
        self.n_slots = 1 << slot_bits
        self.hbits = slot_bits if hash_bits is None else hash_bits
        # store.py Config.hash_matrix: RandomState(seed).randint(0, 2, size=(hbits, NMAX*ID_BITS+LEN_BITS))
        rng = np.random.RandomState(self.seed)
        self.hash_matrix = rng.randint(0, 2, size=(self.hbits, NMAX * ID_BITS + LEN_BITS))

    def slot_of(self, key):
        v = np.array(key_bits(key))
        return int(sum(int(b) << j for j, b in enumerate(self.hash_matrix @ v % 2)))


def name_key_from_tokens(name_toks):
    """store.py StoreScan.name_key. The capture sees the name tokens, then the newline that
    ends the header-to-value line, then the closer's '</' (both captured). Length =
    max(min(n_name, NMAX+2) - 1, 0) with n_name = len(name)+2; the key's tokens are the first
    NMAX captured tokens, zero-padded."""
    captured = list(name_toks) + [Q_NL, Q_LTSLASH]
    n_name = len(name_toks) + 2
    n = max(min(n_name, NMAX + 2) - 1, 0)
    toks = captured[:NMAX]
    toks = toks + [0] * (NMAX - len(toks))
    return (n, tuple(toks))


def stored_len(name_toks):
    """Tokens of the name as the directory stores them (name[:min(length, NMAX)])."""
    return name_toks


class Directory:
    """store.py directory with linear probing (respond(): the SP loop). A slot maps to
    (key, alive). Tombstones keep their key (alive=False) and are walked past."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = {}  # slot -> [key, alive]
        self.order = []

    def probe(self, key):
        """Return (slot, probes, found) for a lookup, exactly the SP loop of respond()."""
        s = self.cfg.slot_of(key)
        probes = 0
        while True:
            probes += 1
            e = self.dir.get(s)
            if e is None or e[0] == key:
                break
            s = (s + 1) % self.cfg.n_slots
        found = e is not None and e[0] == key and e[1]
        return s, probes, found

    def put(self, key):
        """A put/create of a new name (no delete). Returns (probes, status) where status is
        'created', 'replaced' (same key already live: overwrite) or 'full'."""
        s, probes, found = self.probe(key)
        if s in self.dir and self.dir[s][0] == key:
            self.dir[s][1] = True
            return probes, "replaced"
        if len(self.order) >= self.cfg.n_names:
            return probes, "full"
        self.dir[s] = [key, True]
        self.order.append(s)
        return probes, "created"


def call_tokens(op, name=None, text=None):
    """store.py StoreSession.call_tokens (the tokens Qwen writes for a store call)."""
    t = [Q_TCO, Q_NL, Q_LT, Q_FUNCTION]
    t += [OP_JOINED[op]] if op in OP_JOINED else []
    t += [Q_GT, Q_NL]
    for p, val in (("name", name), ("text", text)):
        if val is None:
            continue
        t += [Q_LT, Q_PARAMETER, PARAM_JOINED[p], Q_GT, Q_NL, *val, Q_NL, Q_LTSLASH,
              Q_PARAMETER, Q_GT, Q_NL]
    return t + [Q_LTSLASH, Q_FUNCTION, Q_GT, Q_NL, Q_TCC]


def get_response_tokens(content):
    """respond() get: header (NL, TRO, NL), the chunk's text, then the tail."""
    return [Q_NL, Q_TRO, Q_NL] + list(content) + TAIL


def list_response_tokens(names):
    """respond() list: header, each live name's stored tokens (name[:min(length, NMAX)],
    length includes its newline), then the tail."""
    out = [Q_NL, Q_TRO, Q_NL]
    for nm in names:
        out += nm
    return out + TAIL


def read_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_section(key, value):
    d = read_results()
    d[key] = value
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
