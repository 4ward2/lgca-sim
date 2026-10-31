"""A5 delay-line task (design_A5_v2.md item 3): y_v = x_{v-d} for v >= d, y_v = 0 for v < d, on a chain of L cells.

Output format follows tasks_mul.py: x (B, 1, L) float32 [one channel]; y (B, L) float32; mask all True; diff (B,).
diff = d + 2: the left-to-right information horizon of the K = d+2 set-valued construction (Fable review 4A);
it is the base of the training horizon T = ceil(u * diff), u ~ U(1, 2).  The halting budget is T_max = 4L, not diff.
"""
import numpy as np


def delay_targets(x, d):
    """x (B, L) 0/1 -> y (B, L) with y[:, v] = x[:, v-d] (0 for v < d)."""
    y = np.zeros_like(x)
    if d < x.shape[1]:
        y[:, d:] = x[:, :x.shape[1] - d]
    return y


def generate(rng, B, L, d):
    x = rng.integers(0, 2, size=(B, L)).astype(np.float32)
    y = delay_targets(x, d)
    return dict(x=x[:, None, :], y=y, mask=np.ones(y.shape, bool), diff=np.full(B, d + 2, np.int64))


def ref_delay(xs, d):
    """Pure-Python reference for a single bit list."""
    return [xs[v - d] if v >= d else 0 for v in range(len(xs))]
