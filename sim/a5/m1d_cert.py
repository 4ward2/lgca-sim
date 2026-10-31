"""M1D (chain multiplication) impossibility certificate, Fable review sections 1.4-1.5.

Layout (tasks_mul.gen_m1d): cells 0..n-1 carry (a_i, b_i), cells n..2n-1 carry (0,0); y_i = bit i of a*b.
Codes (c = 2): bit0 = a, bit1 = b, bit2 = in-bounds; virtual cells code 0.  Tables: sim/a5/ac.py format, nin = 8.

Theorem 2.  Fix ANY table set with K latent values.  For n >= K+3 and a = 2^(n-1), the upper half of the outputs
(y_n..y_{2n-1}) is a function of (x_{n-1}, L_{n-1}) alone (L = left-scan set, the right sets depend only on the fixed
zero fill), i.e. of at most 2*2^K keys, while the correct upper halves b>>1 take 2^(n-1) values.  Hence at most
2^(K+2) of the 2^n values of b are fully correct:  exact-match on the family <= 2^(K+2-n).

pigeonhole_counterexample() is the deterministic certificate: it finds two b with equal key and different b>>1,
simulates both with ac.py (and the frozen TernaryNet) and returns one that is wrong.  It cannot fail for n = K+3.
"""
import numpy as np

from . import ac, chain_utils as cu
from .tasks_mul import bits as int_bits


def code(ai, bi):
    return ai | (bi << 1) | 4


def word(n, a, b):
    """The 2n input codes of M1D for operands a, b (< 2^n)."""
    w = np.full(2 * n, code(0, 0), np.int64)
    for i in range(n):
        w[i] = code((a >> i) & 1, (b >> i) & 1)
    return w


def truth(n, a, b):
    return np.array(int_bits(a * b, 2 * n), np.int64)


def family_words(n):
    """a = 2^(n-1), all b in [0, 2^n): codes (2^n, 2n) and true product bits."""
    a = 1 << (n - 1)
    b = np.arange(1 << n)
    X = np.full((1 << n, 2 * n), code(0, 0), np.int64)
    for i in range(n):
        X[:, i] = code((a >> i) & 1, (b >> i) & 1)
    prod = a * b
    Y = (prod[:, None] >> np.arange(2 * n)[None, :]) & 1
    return X, Y


def family_exact_match(tables, n):
    """Exact-match accuracy (all 2n bits right, fixed-point readout) on a = 2^(n-1), all b."""
    X, Y = family_words(n)
    pred = cu.predict(tables, X)
    ok = (pred == Y).all(1)
    return dict(n=n, K=tables["u"].shape[1], correct=int(ok.sum()), total=int(len(ok)), acc=float(ok.mean()))


def family_bound(K, n):
    return 2.0 ** (K + 2 - n)


def check_theorem2(tables, n):
    """Assert the bound on the family; applies for n >= K+3 (below that the bound exceeds 1 and is vacuous)."""
    res = family_exact_match(tables, n)
    K = res["K"]
    res["bound"] = family_bound(K, n)
    assert res["correct"] <= 2 ** (K + 2), f"Theorem 2 violated: {res}"
    assert res["acc"] <= res["bound"] + 1e-12, f"Theorem 2 violated: {res}"
    return res


def pigeonhole_counterexample(tables, n=None):
    """Deterministic FAIL certificate at n = K+3 (default).  Returns dict(n, a, b, word, pred, want, wrong, partner,
    simulator_agrees).  Memory/time: 2^n words of 2n cells; fine up to K ~ 16 (n = 19)."""
    K = tables["u"].shape[1]
    n = n or K + 3
    assert n >= K + 3, "pigeonhole needs n >= K+3"
    a = 1 << (n - 1)
    X, Y = family_words(n)
    L = cu.left_scan(tables["u"], tables["r"], X[:, :n])[:, n - 1]
    b = np.arange(1 << n)
    key = (L << 3) | X[:, n - 1]
    hi = b >> 1
    order = np.lexsort((hi, key))
    ks, hs = key[order], hi[order]
    # group boundaries; a group with differing hi has min != max
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], len(ks)]
    pair = None
    for s, e in zip(starts, ends):
        if hs[s] != hs[e - 1]:
            pair = (int(order[s]), int(order[e - 1]))  # smallest and largest hi in the class
            break
    assert pair is not None, "no key collision: impossible for n >= K+3"
    for bb in pair:
        w = X[bb]
        pred = cu.predict(tables, w[None])[0]
        if not np.array_equal(pred, Y[bb]):
            tern = cu.ternary_predict(tables, w)
            return dict(n=n, K=K, a=a, b=int(bb), word=w.tolist(), pred=pred.tolist(), want=Y[bb].tolist(),
                        wrong=np.flatnonzero(pred != Y[bb]).tolist(), partner=int(pair[0] + pair[1] - bb),
                        simulator_agrees=bool(np.array_equal(tern, pred)), classes=int(len(starts)))
    raise AssertionError("both b of a key collision correct: contradicts Theorem 2 (bug or non-AC semantics)")


def smallest_counterexample(tables, n_scan_max=10):
    """Smallest n <= min(K+3, n_scan_max) (family a = 2^(n-1)) on which some b is wrong, else the pigeonhole
    certificate at n = K+3.  Always returns a simulator-confirmed wrong input of size n <= K+3 (when K+3 > n_scan_max
    and no earlier failure is found the pigeonhole n = K+3 is used)."""
    K = tables["u"].shape[1]
    for n in range(1, min(K + 3, n_scan_max) + 1):
        X, Y = family_words(n)
        pred = cu.predict(tables, X)
        bad = np.flatnonzero((pred != Y).any(1))
        if len(bad):
            b = int(bad[0])
            tern = cu.ternary_predict(tables, X[b])
            return dict(n=n, K=K, a=1 << (n - 1), b=b, word=X[b].tolist(), pred=pred[b].tolist(), want=Y[b].tolist(),
                        wrong=np.flatnonzero(pred[b] != Y[b]).tolist(), simulator_agrees=bool(np.array_equal(tern, pred[b])),
                        method="scan")
    cert = pigeonhole_counterexample(tables)
    cert["method"] = "pigeonhole"
    return cert


def random_tables(rng, K, dens=0.7, nin=8):
    return dict(u=(rng.random((nin, K)) < 0.9).astype(np.uint8),
                r=(rng.random((1, nin, nin, K, K)) < dens).astype(np.uint8),
                o=(rng.random((nin, 1 << K)) < 0.5).astype(np.uint8))
