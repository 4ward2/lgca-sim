"""A5 multiplication task generators (design_A5.md).

M1D: chain of 2n cells, cell i < n carries (a_i, b_i), cells n..2n-1 carry (0, 0). y[i] = bit i of a*b (LSB first).
     x (B, 2, 2n) float32 channels [a, b]; y (B, 2n); mask all True; diff = n.
M2D: n rows x 2n columns. Cell (j, k) carries channels [b_j, a_{k-j}] (0 when k-j is outside 0..n-1).
     s[j, k] = bit k of the running sum P_j = a * (b mod 2^(j+1)); the cell rule is
        s[j,k] = s[j-1,k] ^ (a_{k-j} & b_j) ^ c[j,k-1],   c[j,k] = maj(s[j-1,k], a_{k-j} & b_j, c[j,k-1])
     with s[-1,:] = 0, c[j,-1] = 0.
     x (B, 2, n, 2n); y (B, n, 2n) = s (all per-cell targets); mask (B, n, 2n) = last row only (the task output);
     mask_all = every cell; carry (B, n, 2n) = c; diff = longest run of consecutive 1-carries along any row (>= 1).
Bit lists are LSB first.  The in-bounds ("valid") bit is added later by logic.to_internal.
"""
import itertools

import numpy as np


# ---------------------------------------------------------------- pure-Python references
def bits(v, w):
    return [(v >> i) & 1 for i in range(w)]


def ref_product_bits(a, b, n):
    """Bits (LSB first, 2n of them) of a*b via integer multiplication."""
    return bits(a * b, 2 * n)


def ref_m2d(a, b, n):
    """Per-cell targets from integer arithmetic only. Returns (x[2][n][2n], s[n][2n], c[n][2n]) as nested lists."""
    W = 2 * n
    x = [[[0] * W for _ in range(n)] for _ in range(2)]
    s = [[0] * W for _ in range(n)]
    c = [[0] * W for _ in range(n)]
    prev = 0
    for j in range(n):
        bj = (b >> j) & 1
        A = (a << j) if bj else 0
        cur = prev + A
        carries = prev ^ A ^ cur  # bit k = carry INTO position k
        for k in range(W):
            x[0][j][k] = bj
            x[1][j][k] = (a >> (k - j)) & 1 if 0 <= k - j < n else 0
            s[j][k] = (cur >> k) & 1
            c[j][k] = (carries >> (k + 1)) & 1  # carry OUT of position k
        prev = cur
    return x, s, c


def local_rule_m2d(x):
    """Evaluate the cell rule on a single (2, n, 2n) input purely locally (no integer multiply). Returns (s, c) lists."""
    n, W = len(x[0]), len(x[0][0])
    s = [[0] * W for _ in range(n)]
    c = [[0] * W for _ in range(n)]
    for j in range(n):
        for k in range(W):
            su = s[j - 1][k] if j else 0
            cl = c[j][k - 1] if k else 0
            p = x[0][j][k] & x[1][j][k]
            s[j][k] = su ^ p ^ cl
            c[j][k] = (su & p) | (su & cl) | (p & cl)
    return s, c


# ---------------------------------------------------------------- batch builders
def _from_ab(task, A, Bv, n):
    A = [int(v) for v in A]
    Bv = [int(v) for v in Bv]
    if task == "M1D":
        x = np.zeros((len(A), 2, 2 * n), np.float32)
        y = np.zeros((len(A), 2 * n), np.float32)
        for i, (a, b) in enumerate(zip(A, Bv)):
            x[i, 0, :n] = bits(a, n)
            x[i, 1, :n] = bits(b, n)
            y[i] = ref_product_bits(a, b, n)
        return dict(x=x, y=y, mask=np.ones(y.shape, bool), diff=np.full(len(A), n, np.int64),
                    a=np.array(A, np.int64), b=np.array(Bv, np.int64))
    x = np.zeros((len(A), 2, n, 2 * n), np.float32)
    s = np.zeros((len(A), n, 2 * n), np.float32)
    c = np.zeros((len(A), n, 2 * n), np.float32)
    diff = np.ones(len(A), np.int64)
    for i, (a, b) in enumerate(zip(A, Bv)):
        xi, si, ci = ref_m2d(a, b, n)
        x[i], s[i], c[i] = xi, si, ci
        best = 1
        for row in ci:
            run = 0
            for v in row:
                run = run + 1 if v else 0
                best = max(best, run)
        diff[i] = best
    mask = np.zeros(s.shape, bool)
    mask[:, -1, :] = True
    return dict(x=x, y=s, mask=mask, mask_all=np.ones(s.shape, bool), carry=c, diff=diff,
                a=np.array(A, np.int64), b=np.array(Bv, np.int64))


def generate(task, rng, B, n):
    """Uniform random a, b in [0, 2^n)."""
    A = rng.integers(0, 1 << n, size=B)
    Bv = rng.integers(0, 1 << n, size=B)
    return _from_ab(task, A, Bv, n)


def gen_m1d(rng, B, n):
    return generate("M1D", rng, B, n)


def gen_m2d(rng, B, n):
    return generate("M2D", rng, B, n)


def generate_all(task, n):
    """All 4^n (a, b) pairs, a major."""
    pairs = list(itertools.product(range(1 << n), repeat=2))
    return _from_ab(task, [p[0] for p in pairs], [p[1] for p in pairs], n)
