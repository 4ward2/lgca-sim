"""Delay-line task y_v = x_{v-d} (0 for v < d) on a chain (Fable review section 4A).

  * generator:           words(n) / gen(...)       input bit in channel 0, code = bit | (1 << c)
  * K = d+2 construction: build(d)                  (subset-as-bit-vector encoding, ported from fable_scratch/delay.py)
  * lower bound K >= d-1: pigeonhole_counterexample (any table with K <= d-2 fails on a word of length 2d, found
                          deterministically and confirmed with ac.py and the frozen simulator)
  * SAT entry:           sat_instances / main       (uses k2_sat; K_min pinning for d = 2, 3 on the GitHub runner)

Construction (K = d+2, latent H = {0..d} + {star}; star = d+1).  L_v = {star} + {i <= d : x_{v-i} = 1}:
  u[x](0) = x_bit; all other u = 1;  r[xa,xb](g,h) = [h in (0,star)] if xa is virtual; 1 if xb is virtual;
  else [h == star or h == 0 or (g <= d-1 and h == g+1)];  o[x](S) = [d in S].
star is always present so right-hand support is trivial (R_v = U(x_v) contains L_v).
"""
import argparse
import itertools
import sys

import numpy as np

from . import ac, chain_utils as cu


def code_of(bit, c=1):
    return bit | (1 << c)


def words(n, c=1):
    """All 2^n input words as (codes (2^n, n), bits (2^n, n)); word index w: bit i = (w >> i) & 1."""
    w = np.arange(1 << n)
    bits = (w[:, None] >> np.arange(n)[None, :]) & 1
    return bits | (1 << c), bits


def delay_target(bits, d):
    y = np.zeros_like(bits)
    if d < bits.shape[1]:
        y[:, d:] = bits[:, :bits.shape[1] - d]
    return y


def gen(rng, B, n, d, c=1):
    bits = rng.integers(0, 2, size=(B, n))
    return bits | (1 << c), delay_target(bits, d)


def build(d, c=1):
    """K = d+2 table set in ac.py format (nin = 2^(c+1); real codes 2^c + bit)."""
    K, star, nin = d + 2, d + 1, 1 << (c + 1)
    u = np.ones((nin, K), np.uint8)
    r = np.zeros((1, nin, nin, K, K), np.uint8)
    o = np.zeros((nin, 1 << K), np.uint8)
    for x in range(nin):
        if not (x & 1):
            u[x, 0] = 0
    for xa in range(nin):
        for xb in range(nin):
            for g in range(K):
                for h in range(K):
                    if xa == 0:
                        r[0, xa, xb, g, h] = int(h in (0, star))
                    elif xb == 0:
                        r[0, xa, xb, g, h] = 1
                    else:
                        r[0, xa, xb, g, h] = int(h == star or h == 0 or (g <= d - 1 and h == g + 1))
    for x in range(nin):
        for S in range(1 << K):
            o[x, S] = (S >> d) & 1
    return dict(u=u, r=r, o=o)


def verify_construction(d, n_exh=7, n_rand=(8, 12), n_samples=60, seed=200201):
    """Returns dict(words=..., wrong=...): exhaustive n = 1..n_exh, random lengths n_rand[0]..n_rand[1]."""
    T = build(d)
    tot = bad = 0
    for n in range(1, n_exh + 1):
        X, bits = words(n)
        pred = cu.predict(T, X)
        tot += len(X)
        bad += int((pred != delay_target(bits, d)).any(1).sum())
    rng = np.random.default_rng(seed + d)
    for n in range(n_rand[0], n_rand[1] + 1):
        X, y = gen(rng, n_samples, n, d)
        pred = cu.predict(T, X)
        tot += len(X)
        bad += int((pred != y).any(1).sum())
    return dict(d=d, K=d + 2, words=tot, wrong=bad)


def pigeonhole_counterexample(tables, d, c=1):
    """Fable section 4A lower bound.  Words w (d bits) followed by d zeros; y_{d..2d-1} = w.  Those outputs depend only
    on (x_{d-1}, L_{d-1}) (the suffix is fixed), so two different w with the same key cannot both be right.  With K latent
    values there are <= 2*2^K keys, so K <= d-2 (2^d > 2^(K+1)) always yields a collision.
    Returns a counterexample dict (word codes, pred, want, simulator-confirmed), or None if no key collision exists
    (possible only when K >= d-1)."""
    u, r, o = tables["u"], tables["r"], tables["o"]
    X, bits = words(d, c)
    key_x = X[:, d - 1]
    L = cu.left_scan(u, r, X)[:, d - 1]
    key = (L << 3) | key_x  # key_x < 2^(c+1) <= 8 for c <= 2
    order = np.argsort(key, kind="stable")
    ks = key[order]
    first = {}
    pair = None
    for idx, k in zip(order.tolist(), ks.tolist()):
        if k in first:
            pair = (first[k], idx)
            break
        first[k] = idx
    if pair is None:
        return None
    full = np.concatenate([X, np.full((len(X), d), code_of(0, c), np.int64)], 1)
    for w in pair:
        word = full[w]
        pred = cu.predict(tables, word[None])[0]
        want = delay_target(np.concatenate([bits[w], np.zeros(d, np.int64)])[None], d)[0]
        if not np.array_equal(pred, want):
            tern = cu.ternary_predict(tables, word)
            return dict(d=d, K=u.shape[1], word=word.tolist(), pred=pred.tolist(), want=want.tolist(),
                        confirmed_by_simulator=bool(np.array_equal(tern, pred)), partner=int(pair[0] + pair[1] - w),
                        pair=pair, n_keys=int(len(first)))
    raise AssertionError("pigeonhole: both words of a key collision are correct (theory violated)")


# ------------------------------------------------------------------------------------------------ SAT entry
def sat_instances(d, nmax, c=1):
    """All words of lengths 1..nmax, every cell scored.  Format consumed by k2_sat.build_cnf."""
    inst = []
    for n in range(1, nmax + 1):
        X, bits = words(n, c)
        y = delay_target(bits, d)
        for i in range(len(X)):
            inst.append(dict(xi=X[i], tgt=y[i], mask=np.ones(n, bool)))
    return inst


def main(argv=None):
    ap = argparse.ArgumentParser(description="delay-line d: SAT 'exists K-latent table set correct on all words n<=nmax'")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--n", type=int, default=None, help="max word length (default 2d+1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-limit", type=float, default=0)
    ap.add_argument("--solver", default="cadical153")
    a = ap.parse_args(argv)
    from . import k2_sat
    k2_sat.cli(["--task", "delay", "--d", str(a.d), "--K", str(a.K), "--n", str(a.n or 2 * a.d + 1)]
               + (["--out", a.out] if a.out else []) + ["--time-limit", str(a.time_limit), "--solver", a.solver])


if __name__ == "__main__":
    main()
