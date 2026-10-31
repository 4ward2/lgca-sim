"""General-K synchronous arc-consistency interpreter in TernaryNet's table form (numpy, no torch).

Tables (bool/0-1 arrays):  u (nin, K);  r (dims, nin, nin, K, K) with r[axis][x_first, x_second][h_first, h_second]
where "first" is the cell with the smaller coordinate along that axis;  o (nin, 2^K) readout indexed by the whole set,
subset index = sum_h 2^h * [h in S].  Input code: bit k = channel k, top bit = in-bounds (valid) bit; virtual
out-of-bounds cells have code 0 and a full domain that is never updated.
Update (identical to b_arm.TernaryNet.update):  c'_h(i) = u[x_i](h) AND AND_d OR_h' ( r_d(h', h) AND c_h'(j_d) ).
"""
import numpy as np


def codes_from_x(x):
    """x (B, c, *sp) 0/1 array -> (B, *sp) int codes with the valid bit appended on top."""
    x = np.asarray(x) > 0.5
    c = x.shape[1]
    code = np.zeros((x.shape[0],) + x.shape[2:], np.int64)
    for k in range(c):
        code += x[:, k].astype(np.int64) << k
    return code + (1 << c)


def update(u, r, c, xi):
    """One synchronous step. c: (B, *sp, K) bool; xi: (B, *sp) int. Works for dims 1 and 2."""
    dims = xi.ndim - 1
    K = c.shape[-1]
    cp = np.pad(c, [(0, 0)] + [(1, 1)] * dims + [(0, 0)], constant_values=True)
    xp = np.pad(xi, [(0, 0)] + [(1, 1)] * dims)
    out = u[xi].astype(bool)
    if dims == 1:
        N = c.shape[1]
        me = xp[:, 1:N + 1]
        nb = [(cp[:, 0:N], r[0][xp[:, 0:N], me]),
              (cp[:, 2:N + 2], np.swapaxes(r[0][me, xp[:, 2:N + 2]], -1, -2))]
    else:
        H, W = c.shape[1], c.shape[2]
        me = xp[:, 1:H + 1, 1:W + 1]
        nb = [(cp[:, 0:H, 1:W + 1], r[0][xp[:, 0:H, 1:W + 1], me]),
              (cp[:, 2:H + 2, 1:W + 1], np.swapaxes(r[0][me, xp[:, 2:H + 2, 1:W + 1]], -1, -2)),
              (cp[:, 1:H + 1, 0:W], r[1][xp[:, 1:H + 1, 0:W], me]),
              (cp[:, 1:H + 1, 2:W + 2], np.swapaxes(r[1][me, xp[:, 1:H + 1, 2:W + 2]], -1, -2))]
    for cj, R in nb:
        out = out & (R.astype(bool) & cj[..., :, None]).any(axis=-2)
    return out


def fixed_point(u, r, xi, tmax=10000):
    """Kleene iteration from full domains. Returns (c, steps)."""
    K = u.shape[1]
    c = np.ones(xi.shape + (K,), bool)
    for t in range(tmax):
        cn = update(u, r, c, xi)
        if np.array_equal(cn, c):
            return c, t
        c = cn
    return c, tmax


def subset_index(c):
    K = c.shape[-1]
    return (c.astype(np.int64) << np.arange(K)).sum(-1)


def readout(o, c, xi):
    return o[xi, subset_index(c)]
