"""Task generators for Stage A (see brief_A.md section 1).

Every generator returns a dict of numpy arrays:
  x     : (B, c, *sp) float32 task inputs (without the in-bounds bit)
  y     : (B, *sp)    float32 per-position labels
  diff  : (B,)        int     difficulty D
Spatial layout is (N,) for 1D tasks and (H, W) for 2D.
"""
import numpy as np

TASKS = {
    # name: (dims, c, train size range, in-dist size, extrapolation sizes)
    "P": dict(dims=1, c=1, train=(4, 16), indist=16, extra=(32, 64, 128, 256), bar=64),
    "A": dict(dims=1, c=2, train=(4, 16), indist=16, extra=(32, 64, 128, 256), bar=64),
    "M": dict(dims=2, c=2, train=(6, 12), indist=12, extra=(16, 24, 32), bar=32),
}
TEST_SEED = 12345
VAL_SEED = 777
N_TEST = {"P": 1000, "A": 1000, "M": 500}
N_VAL = {"P": 1000, "A": 1000, "M": 500}
WALL_P = 0.4
FULLCHAIN_P = 0.25


def gen_parity(rng, B, n):
    x = rng.integers(0, 2, size=(B, n)).astype(np.int64)
    y = np.cumsum(x, axis=1) % 2
    return dict(x=x[:, None, :].astype(np.float32), y=y.astype(np.float32),
                diff=np.full(B, n, dtype=np.int64))


def _carry_chain(a, b):
    """a, b: (B, L) bits, LSB first. Returns sum bits (B, L) and longest carry run + 1."""
    B, L = a.shape
    s = np.zeros_like(a)
    carry = np.zeros(B, dtype=np.int64)
    run = np.zeros(B, dtype=np.int64)
    best = np.zeros(B, dtype=np.int64)
    for i in range(L):
        t = a[:, i] + b[:, i] + carry
        s[:, i] = t & 1
        run = np.where(carry == 1, run + 1, 0)
        best = np.maximum(best, run)
        carry = t >> 1
    return s, best + 1


def gen_addition(rng, B, n, fullchain_p=FULLCHAIN_P, force_fullchain=False):
    """n-bit addition with long carry-chain stress construction."""
    a = rng.integers(0, 2, size=(B, n)).astype(np.int64)
    b = rng.integers(0, 2, size=(B, n)).astype(np.int64)
    use = np.ones(B, dtype=bool) if force_fullchain else (rng.random(B) < fullchain_p)
    lengths = np.zeros(B, dtype=np.int64)
    for j in np.flatnonzero(use):
        s = int(rng.integers(0, max(1, n - (n+1)//2 + 1)))
        L = int(rng.integers((n+1)//2, n-s+1))
        lengths[j] = L
        a[j, s] = b[j, s] = 1
        for i in range(s+1, s+L):
            if rng.integers(2): a[j,i], b[j,i] = 1, 0
            else: a[j,i], b[j,i] = 0, 1
    pad = np.zeros((B, 1), dtype=np.int64)
    a1 = np.concatenate([a, pad], 1); b1 = np.concatenate([b, pad], 1)
    s, d = _carry_chain(a1, b1)
    return dict(x=np.stack([a1,b1],1).astype(np.float32), y=s.astype(np.float32), diff=d)


def _bfs_batch(open_, src):
    """Vectorised BFS by repeated 4-neighbour dilation.
    open_, src: (B, H, W) bool. Returns reach (bool) and eccentricity of src (int)."""
    reach = src & open_
    ecc = np.zeros(open_.shape[0], dtype=np.int64)
    step = 0
    while True:
        step += 1
        d = reach.copy()
        d[:, 1:, :] |= reach[:, :-1, :]
        d[:, :-1, :] |= reach[:, 1:, :]
        d[:, :, 1:] |= reach[:, :, :-1]
        d[:, :, :-1] |= reach[:, :, 1:]
        d &= open_
        grew = (d != reach).any(axis=(1, 2))
        if not grew.any():
            break
        ecc = np.where(grew, step, ecc)
        reach = d
    return reach, ecc


def gen_maze(rng, B, n, wall_p=WALL_P):
    open_ = rng.random((B, n, n)) >= wall_p
    # make sure every grid has at least one open cell
    empty = ~open_.reshape(B, -1).any(1)
    open_[empty, 0, 0] = True
    src = np.zeros_like(open_)
    flat = open_.reshape(B, -1)
    for i in range(B):
        idx = np.flatnonzero(flat[i])
        k = idx[rng.integers(len(idx))]
        src[i].flat[k] = True
    reach, ecc = _bfs_batch(open_, src)
    x = np.stack([open_, src], 1).astype(np.float32)
    # D = BFS eccentricity of the source within its component (brief_A.md section 1); astra review fix
    return dict(x=x, y=reach.astype(np.float32), diff=ecc)


def generate(task, rng, B, n, **kw):
    if task == "P":
        return gen_parity(rng, B, n)
    if task == "A":
        return gen_addition(rng, B, n, **kw)
    if task == "M":
        return gen_maze(rng, B, n, **kw)
    raise ValueError(task)


def train_size(task, rng):
    lo, hi = TASKS[task]["train"]
    return int(rng.integers(lo, hi + 1))


def difficulty_quantile(task, q=0.99, samples=20000, seed=0):
    """Quantile of D over the training distribution (from the generator only)."""
    rng = np.random.default_rng(seed)
    ds = []
    per = 500
    for _ in range(samples // per):
        ds.append(generate(task, rng, per, train_size(task, rng))["diff"])
    return float(np.quantile(np.concatenate(ds), q))


def fixed_set(task, n, count, seed, **kw):
    rng = np.random.default_rng([seed, n, 1 if kw.get("force_fullchain") else 0])
    return generate(task, rng, count, n, **kw)
