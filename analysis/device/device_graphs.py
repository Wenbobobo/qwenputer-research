"""Tiny dedicated "devices" built in the q35 IR (Qwenputer stage-3 idea B prototype).

Two devices, each a standalone token family + circuit, in the same conventions as chip8_q35:
  * every token carries an input code feature CODE_AMP * (2 bit - 1) (tied lm_head reads it back);
  * the last sublayer is the *exit*: it clears the input code and writes KAPPA * (2 bit - 1) of
    the next token's code where the device owns the position (else 0); emit.emit_layers with
    keep then clears every other chip dim and the ballast.

ADD8   record  A(a) B(b) S(a+b)           tokens A 0..255, B 256..511, S 512..1023 (9-bit sum)
       circuit L1 attn: previous-token head (a's bits at the B token, conv lag 1)
               L1 mlp : (input-code clear only)
               L2 mlp : carry-lookahead adder folded into the exit (c_i = [a mod 2^i + b mod 2^i
                        >= 2^i] is one threshold on a linear form; s_i = a_i + b_i + c_i - 2c_{i+1})
               (the adder itself fits in L1 mlp; the extra layer is only the I/O code convention)
SORT4  record  X0 X1 X2 X3 Y0 Y1 Y2 Y3    tokens Xk = k*256 + v, Yk = (4+k)*256 + v (8-bit values)
       circuit L1 attn: snapshot head, written at X3 from conv lags 0..3 (all four inputs)
               L1 mlp : 6 comparators d_ij = [x_i <= x_j]; ranks are linear in them; code clear
               L2 mlp : exit = select x_i with rank_i == k for the output slot k
"""

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Chan, Target
from chipformer.q35.prims import Circ

CODE_AMP = 128
KAPPA = 2.0**-20


class Device(Circ):
    """Common token plumbing: family/payload code features."""

    def __init__(self, nfam_bits: int, npay: int, family_of, target: Target | None = None):
        super().__init__(target or Target())
        self.nfam_bits, self.npay, self.family_of = nfam_bits, npay, family_of
        self.ncode = nfam_bits + npay
        self.code = [self.token_feature(f"code{i}", lambda t, i=i: CODE_AMP *
                                        (2 * self.token_code(t)[i] - 1))
                     for i in range(self.ncode)]

    def token_code(self, t: int) -> list[int]:
        fam, pay = self.family_of(t)
        return [fam >> i & 1 for i in range(self.nfam_bits)] + \
               [pay >> i & 1 for i in range(self.npay)]

    def exit(self, code_terms: list, owner: Expr, top: int) -> None:
        """code_terms[i]: unit terms whose sum is [next code bit i] (0/1) AND owner. One node per
        code dim: clear the input code, add KAPPA * (2 bit - 1) where owned."""
        one = self.one
        # the input code is cleared one MLP sublayer earlier (as chip8_q35's code.clear): -e and
        # +-KAPPA cannot be summed exactly in one bf16 MLP output
        for i, e in enumerate(self.code):
            self.ffn([(-1, one, e), (1, one, -1 * e)], f"code.clear{i}", into=e)
        top = max(top, max(n.level for n in self.nodes))
        for i, e in enumerate(self.code):
            terms = [(2 * KAPPA * c, a, b) for c, a, b in code_terms[i]]
            terms += [(-KAPPA, one, owner)]
            self.ffn(terms, f"exit.code{i}", into=e, min_level=top)
        self.keep = [next(iter(e.terms)) for e in self.code]

    def decode_out(self, row) -> int | None:
        """Token from KAPPA * +-1 code values (None if not owned)."""
        if all(v == 0 for v in row):
            return None
        bits = [1 if v > 0 else 0 for v in row]
        fam = sum(b << i for i, b in enumerate(bits[: self.nfam_bits]))
        pay = sum(b << i for i, b in enumerate(bits[self.nfam_bits:]))
        return self.token_of(fam, pay)


# --------------------------------------------------------------------------- ADD8

A_BASE, B_BASE, S_BASE = 0, 256, 512


def add8_family(t):
    if t < B_BASE:
        return 0, t - A_BASE
    if t < S_BASE:
        return 1, t - B_BASE
    return 2, t - S_BASE


class Add8(Device):
    def token_of(self, fam, pay):
        return (A_BASE, B_BASE, S_BASE)[fam] + pay


def build_add8(width: int = 8) -> Add8:
    assert width <= 8  # the lookahead threshold's range must stay within 256 gate steps
    c = Add8(2, width + 1, add8_family)
    f, one = c.token_feature, c.one
    abit = [f(f"a{i}", lambda t, i=i: (t - A_BASE) >> i & 1 if t < B_BASE else 0)
            for i in range(width)]
    bbit = [f(f"b{i}", lambda t, i=i: (t - B_BASE) >> i & 1 if B_BASE <= t < S_BASE else 0)
            for i in range(width)]
    is_b = f("is_b", lambda t: int(B_BASE <= t < S_BASE))
    # previous-token head: a's bits seen at the next token (conv tap lag 1)
    a_prev = c.slot_head("prev_a", {0: one}, {0: one},
                         {2 + i: Chan(abit[i], (0, 1, 0, 0)) for i in range(width)}, one)
    owner = is_b

    def low(bits, i):
        return sum((bits[k] * (1 << k) for k in range(i)), Expr())

    def carry_terms(i, coef):  # coef * [carry into bit i] AND owner
        if i == 0:
            return []
        z = low(a_prev, i) + low(bbit, i)
        return c.t_ge(z, 1 << i, 0, (1 << (i + 1)) - 2, coef, lits=[owner])

    code_terms = []
    # family S = 2: bits (0, 1), set exactly where owned
    code_terms.append([])
    code_terms.append([(1, one, owner)])
    for i in range(width):  # s_i = a_i + b_i + c_i - 2 c_{i+1}   (AND owner)
        ts = [(1, a_prev[i], owner), (1, bbit[i], owner)]
        ts += carry_terms(i, 1) + carry_terms(i + 1, -2)
        code_terms.append(ts)
    code_terms.append(carry_terms(width, 1))  # carry out = bit 'width'
    c.exit(code_terms, owner, top=0)
    c.width = width
    return c


def add8_record(a: int, b: int) -> list[int]:
    return [A_BASE + a, B_BASE + b, S_BASE + a + b]


# --------------------------------------------------------------------------- SORT4


def sort4_family(t):
    return t // 256, t % 256


class Sort4(Device):
    def token_of(self, fam, pay):
        return fam * 256 + pay


def build_sort4() -> Sort4:
    c = Sort4(3, 8, sort4_family)
    f, one = c.token_feature, c.one
    xbit = [f(f"x{i}", lambda t, i=i: t % 256 >> i & 1 if t < 1024 else 0) for i in range(8)]
    pos = [f(f"is_fam{k}", lambda t, k=k: int(t // 256 == k)) for k in range(8)]
    is_x3 = pos[3]
    # snapshot head written at X3: x_j sits at conv lag 3 - j
    data = {}
    for j in range(4):
        taps = tuple(1.0 if lag == 3 - j else 0.0 for lag in range(4))
        for b in range(8):
            data[2 + 8 * j + b] = Chan(xbit[b], taps)
    out = c.slot_head("snap", {0: is_x3}, {0: one}, data, is_x3)
    x = [out[8 * j: 8 * j + 8] for j in range(4)]
    val = [sum((x[j][b] * (1 << b) for b in range(8)), Expr()) for j in range(4)]
    # comparators: d[i, j] = [x_i <= x_j] for i < j
    d = {}
    for i in range(4):
        for j in range(i + 1, 4):
            d[i, j] = c.node(c.t_ge(val[j] - val[i], 0, -255, 255), f"cmp{i}{j}")
    # stable rank: #{j < i: x_j <= x_i} + #{j > i: x_j < x_i}
    rank = []
    for i in range(4):
        r = Expr()
        for j in range(4):
            if j < i:
                r = r + d[j, i]
            elif j > i:
                r = r + (one - d[i, j])
        rank.append(r)
    # output slot k is predicted at token X3 (k=0), Y0 (k=1), Y1 (k=2), Y2 (k=3)
    slot = [pos[3], pos[4], pos[5], pos[6]]
    owner = sum(slot, Expr())
    code_terms = []
    for fb in range(3):  # next family = 4 + k
        code_terms.append([(1, one, slot[k]) for k in range(4) if (4 + k) >> fb & 1])
    for b in range(8):
        ts = []
        for k in range(4):
            for i in range(4):
                ts += c.t_eq(rank[i], k, 0, 3, 1, lits=[slot[k], x[i][b]])
        code_terms.append(ts)
    c.exit(code_terms, owner, top=c.level(*rank))
    return c


def sort4_record(xs: list[int]) -> list[int]:
    ys = sorted(xs)
    return [k * 256 + v for k, v in enumerate(xs)] + [(4 + k) * 256 + v for k, v in enumerate(ys)]


# --------------------------------------------------------------------------- DADD (digit-serial)
# Decimal bignum add, least significant digit first, interleaved:
#   P(a_k, b_k) S(z_k) P(a_k+1, b_k+1) S(z_k+1) ...     z_k = a_k + b_k + carry_in in 0..19
# P tokens 0..99 (10 a + b), S tokens 100..119 (z; the printed digit is z mod 10, the carry z >= 10),
# G = 120 starts a record:  G P S P S ... (2n + 1 tokens for n digits).
# The device predicts S at every P token; the carry comes from the previous token (conv lag 1).

P_BASE, SD_BASE = 0, 100
G_TOK = 120  # record start ("+"): its carry-out feature is 0, so a record never sees the last one's


def dadd_family(t):
    return (0, t - P_BASE) if t < SD_BASE else (1, t - SD_BASE)


class DAdd(Device):
    def token_of(self, fam, pay):
        return (P_BASE, SD_BASE)[fam] + pay


def build_dadd() -> DAdd:
    c = DAdd(1, 7, dadd_family)  # payload 7 bits (P needs 0..99)
    f, one = c.token_feature, c.one
    is_p = f("is_p", lambda t: int(t < SD_BASE))
    a = f("pa", lambda t: (t - P_BASE) // 10 if t < SD_BASE else 0)
    b = f("pb", lambda t: (t - P_BASE) % 10 if t < SD_BASE else 0)
    cout = f("s_carry", lambda t: int(SD_BASE + 10 <= t < SD_BASE + 20))  # G (120): carry 0
    (cin,) = c.slot_head("prev_carry", {0: one}, {0: one}, {2: Chan(cout, (0, 1, 0, 0))}, one)
    z = a + b + cin  # 0..19
    owner = is_p
    code_terms = [[(1, one, owner)]]  # next family S = 1
    for i in range(7):
        ts = []
        for v in range(20):
            if v >> i & 1:
                ts += c.t_eq(z, v, 0, 19, 1, lits=[owner])
        code_terms.append(ts)
    c.exit(code_terms, owner, top=0)
    return c


def dadd_record(x: int, y: int) -> list[int]:
    """Digits LSB first over max(len) + 0 positions; the last S carries the final carry."""
    n = max(len(str(x)), len(str(y)))
    toks, carry = [G_TOK], 0  # G: record start (no carry in)
    for k in range(n):
        ak, bk = x // 10**k % 10, y // 10**k % 10
        zk = ak + bk + carry
        toks += [P_BASE + 10 * ak + bk, SD_BASE + zk]
        carry = int(zk >= 10)
    return toks


def dadd_value(toks: list[int]) -> int:
    zs = [t - SD_BASE for t in toks if SD_BASE <= t < G_TOK]
    return sum((z % 10) * 10**k for k, z in enumerate(zs)) + (zs[-1] >= 10) * 10**len(zs)
