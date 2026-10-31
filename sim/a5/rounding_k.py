"""General-K rounding of a trained BPNet to a deployed ternary AC table set in the sim/a5/ac.py format.

Rule (design_A5_v2.md, fixed before training; generalises b_arm.bp_to_ternary, thr = 2):
  r[d][xa,xb][ha,hb] = [Psi - max_{h2} Psi(ha,h2) > -thr]  AND  [Psi - max_{h2} Psi(h2,hb) > -thr]
  u[x][h]            = [U - max_{h2} U(h2) > -thr]
  o[x][{h}]          = [O[x][h] > 0]                          (singleton subsets, by sign of O)
  o[x][S], |S| != 1  = majority label (ties -> 0) over the masked cells of ONE calibration batch whose fixed point
                       (computed with the rounded u, r) is S and whose local input is x;  never observed -> 0.
Returns numpy tables {u, r, o} with o of shape (nin, 2^K) indexed by subset_index.
"""
import numpy as np
import torch

from . import ac  # noqa: F401  (also puts sim/ on sys.path)
from .ac import codes_from_x, fixed_point, subset_index

THR = 2.0


@torch.no_grad()
def round_tables(bp, thr=THR):
    """u, r and singleton o from the BPNet parameters. o for non-singletons is left 0 (see calibrate_o)."""
    K = bp.K
    Psi, U, O = bp.Psi.data, bp.U.data, bp.O.data
    rel = Psi - Psi.amax(dim=-1, keepdim=True)
    rel2 = Psi - Psi.amax(dim=-2, keepdim=True)
    r = ((rel > -thr) & (rel2 > -thr)).numpy().astype(np.uint8)
    u = ((U - U.amax(dim=-1, keepdim=True)) > -thr).numpy().astype(np.uint8)
    o = np.zeros((bp.nin, 1 << K), np.uint8)
    Op = (O > 0).numpy().astype(np.uint8)
    for h in range(K):
        o[:, 1 << h] = Op[:, h]
    return dict(u=u, r=r, o=o)


def calibrate_o(tabs, fit_data, tmax):
    """Majority vote of the label over masked cells of fit_data, grouped by (x, fixed-point set S), for |S| != 1."""
    u, r, o = tabs["u"], tabs["r"], tabs["o"].copy()
    nin, nS = o.shape
    xi = codes_from_x(fit_data["x"])
    c, _ = fixed_point(u, r, xi, tmax)
    s = subset_index(c)
    y = (np.asarray(fit_data["y"]) > 0.5).astype(np.int64)
    mk = np.asarray(fit_data["mask"], bool)
    idx = ((xi * nS + s) * 2 + y)[mk]
    cnt = np.bincount(idx, minlength=nin * nS * 2).reshape(nin, nS, 2)
    pop = np.array([bin(i).count("1") for i in range(nS)])
    vote = (cnt[..., 1] > cnt[..., 0]).astype(np.uint8)
    seen = cnt.sum(-1) > 0
    sel = seen & (pop != 1)[None, :]
    o[sel] = vote[sel]
    return dict(u=u, r=r, o=o), int(seen.sum())


def round_bp(bp, fit_data, tmax, thr=THR):
    """Full deployment rounding. Returns (tables, number of (x, S) pairs observed in the calibration batch)."""
    return calibrate_o(round_tables(bp, thr), fit_data, tmax)


def default_tmax(x_shape):
    """Calibration fixed-point horizon: 4L for chains, 4n for n x 2n grids (x_shape = batch['x'].shape)."""
    return 4 * (x_shape[-1] if len(x_shape) == 3 else x_shape[-2])


# ------------------------------------------------------------------ JSON (o is sparse: only the 1-entries)
def to_json(tabs):
    o = tabs["o"]
    return dict(u=tabs["u"].tolist(), r=tabs["r"].tolist(), o_shape=list(o.shape),
                o_ones=np.argwhere(o > 0).tolist())


def from_json(d):
    o = np.zeros(d["o_shape"], np.uint8)
    for x, s in d["o_ones"]:
        o[x, s] = 1
    return dict(u=np.array(d["u"], np.uint8), r=np.array(d["r"], np.uint8), o=o)
