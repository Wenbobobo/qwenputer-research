"""Numpy check of Gated DeltaNet as RAM (Qwenputer's memory primitive).

Delta rule with gate alpha and write strength beta (per value head):
    S_t = alpha * S_{t-1} + beta * k_t (v_t - S_{t-1}^T k_t)^T      (S: d_k x d_v)
    read:  o_t = S_t^T q_t
With one-hot k = e_i, beta = 1, alpha = 1 the update REPLACES row i of S with v and leaves
every other row untouched, so S is an addressable d_k-row RAM of d_v-wide words. A read with
q = e_i returns row i exactly. This script demonstrates it against a plain dict reference.
Run: python repro/verify_gdn_ram.py
"""
import numpy as np


def gdn_step(S, k, v, alpha=1.0, beta=1.0):
    S = alpha * S
    return S + beta * np.outer(k, v - S.T @ k)


def onehot(i, n):
    e = np.zeros(n)
    e[i] = 1.0
    return e


def main(d_k=128, d_v=128, n_ops=5000, seed=0):
    rng = np.random.default_rng(seed)
    S = np.zeros((d_k, d_v))
    ref = {}                      # address -> word (plain-dict reference RAM)
    writes = reads = 0
    max_err = 0.0
    for _ in range(n_ops):
        addr = int(rng.integers(d_k))
        if rng.random() < 0.5:    # write: integer-valued words (like bit/byte encodings)
            word = rng.integers(-2**15, 2**15, size=d_v).astype(np.float64)
            S_old = S.copy()
            S = gdn_step(S, onehot(addr, d_k), word)
            ref[addr] = word
            others = [r for r in range(d_k) if r != addr]
            assert np.array_equal(S[others], S_old[others]), "write disturbed other rows"
            writes += 1
        else:                     # read with q = e_addr
            got = S.T @ onehot(addr, d_k)
            want = ref.get(addr, np.zeros(d_v))
            max_err = max(max_err, float(np.abs(got - want).max()))
            assert np.array_equal(got, want), f"read mismatch at {addr}"
            reads += 1
    full = np.stack([ref.get(r, np.zeros(d_v)) for r in range(d_k)])
    assert np.array_equal(S, full)
    print(f"d_k={d_k} d_v={d_v}: {writes} writes, {reads} reads, final state == dict RAM: True, "
          f"max read error {max_err}")
    # Counter-examples: why alpha=1, beta=1 and one-hot keys matter.
    S0 = np.stack([onehot(r % d_v, d_v) * (r + 1) for r in range(d_k)])
    i, v = 3, np.full(d_v, 7.0)
    a = gdn_step(S0, onehot(i, d_k), v, alpha=0.9)
    b = gdn_step(S0, onehot(i, d_k), v, beta=0.5)
    k2 = (onehot(i, d_k) + onehot(i + 1, d_k)) / np.sqrt(2)
    c = gdn_step(S0, k2, v)
    print(f"alpha=0.9: row {i} exact {np.array_equal(a[i], v)}, other rows kept "
          f"{np.array_equal(np.delete(a, i, 0), np.delete(S0, i, 0))}")
    print(f"beta=0.5 : row {i} exact {np.array_equal(b[i], v)} (half-way: {b[i][:2]})")
    print(f"non-one-hot k: rows {i},{i+1} disturbed, row {i} exact {np.array_equal(c[i], v)}")
    print("PASS")


if __name__ == "__main__":
    main()
