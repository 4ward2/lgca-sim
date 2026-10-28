"""Stage A2 arm B: learned local belief propagation, deployed as ternary monotone arc consistency
(algorithmic line) or hard max-product LLR-BP (image line). brief_A2.md section 3 (frozen settings).

Consolidated from reviews/a2/fable/{bpnet.py, ternary.py, bp2ternary.py} (Fable, design sections 1, 6, 7).

Model (section 3.1): K=2 hidden variable per cell; three tables shared by all cells, indexed by the local input
x~ = (task bits, in-bounds bit): U[x~](h), Psi_e[x~_a, x~_b](h_a, h_b) (e = edge orientation), O[x~](h).
Training (3.2): sum-product soft BP unrolled T = ceil(u * D_batch) steps, u ~ U(1, 2); masked BCE; Adam 0.03;
batch 128 (1D) / 32 (2D); 2000 steps; evaluation every 250 steps; no damping, no bound.
Deployment (3.3), fixed per task line before training:
  * P/A/M/cycles/cycles_reach: ternary arc consistency. Tables rounded to bits with thr = 2 (no search);
    o(unknown), o(contradiction) by majority vote over the masked cells of ONE training-stream batch.
  * inpaint/life: hard max-product LLR-BP, Q = 6 bits, delta = 0.25 (no rounding to bits).
Checkpoints (3.4): best_deploy (deployed model's selection score on val 777, ties -> earlier step), best_soft
(soft BP evaluated with hard argmax readout, diagnostic), last (step 2000).
Selection score (3.5): mean "halted and correct" over the in-distribution sizes (1D {8,12,16}; 2D {8,10,12};
inpaint (16, d in {2,3,4}); life {8,12,16}; evaluate_a2.VAL_SPECS for the new tasks).
Halting (3.6): sim/evaluate.run_hard (P/A/M) and sim/evaluate_a2.run_masked (new tasks): readout unchanged for
ceil(T_max/2) steps, T_max = 4L (1D) / n^2 (2D); the new tasks halt and score on masked cells only (4b).
Four-state readout (3.8): the readout depends only on the set state ({0}, {1}, unknown, contradiction) and x~.

CLI:
  python sim/b_arm.py train --task {P,A,M,cycles,cycles_reach,inpaint,life} --seed S --out DIR [--threads N]
  python sim/b_arm.py evaluate DIR [--locked] [--threads N]     # --locked writes test.json (test seeds)
  python sim/b_arm.py evaluate DIR --hook                       # same code path on val 777, in-distribution only
  python sim/b_arm.py diag DIR                                  # Q/Q+ small-graph oracle diagnostic (section 5)
  python sim/b_arm.py oracle                                    # hand-built tables through the pipeline
No existing file under sim/ is modified or imported for anything but reading.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tasks as tk  # noqa: E402
import tasks_a2 as t2  # noqa: E402
import evaluate as ev  # noqa: E402
import evaluate_a2 as ea  # noqa: E402
from logic import to_internal  # noqa: E402

A1_TASKS = ("P", "A", "M")
A2_TASKS = ("cycles", "cycles_reach", "inpaint", "life")
TASKS = A1_TASKS + A2_TASKS
DEPLOY = {"P": "ternary", "A": "ternary", "M": "ternary", "cycles": "ternary", "cycles_reach": "ternary",
          "inpaint": "bp_hard", "life": "bp_hard"}
THR = 2.0
Q_BITS, DELTA = 6, 0.25
LR, BETA, STEPS, EVAL_EVERY = 0.03, 2.0, 2000, 250
SIZES_1D, SIZES_2D = (8, 12, 16), (8, 10, 12)
VAL_SEED, VAL2_SEED = tk.VAL_SEED, ev.VAL2_SEED
STATE_NAMES = ("contradiction", "{0}", "{1}", "unknown")  # subset index = c0 + 2*c1


def task_info(task):
    if task in A1_TASKS:
        i = tk.TASKS[task]
        return i["dims"], i["c"]
    i = t2.TASKS_A2[task]
    return i["dims"], i["c"]


def xidx(xv):
    """xv: (c+1, B, *sp) float -> (B, *sp) long index of the local input bits (bit k = channel k)."""
    idx = torch.zeros(xv.shape[1:], dtype=torch.long, device=xv.device)
    for k in range(xv.shape[0]):
        idx = idx + (xv[k] > 0.5).long() * (1 << k)
    return idx


# ====================================================================================== soft / LLR BP
class BPNet(nn.Module):
    """K-state pairwise MRF with input-conditioned tables and synchronous BP (flooding schedule).
    State seen by run_hard/run_masked: channel 0 = readout, then nb*K message values."""

    def __init__(self, dims, c, K=2, Q=Q_BITS, delta=DELTA, infer="sum", init=0.3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dims, self.c, self.K = dims, c, K
        self.nin = 1 << (c + 1)
        self.E = dims
        self.nb = 2 if dims == 1 else 4  # message slots: 1D [from-left, from-right]; 2D [N, S, W, E]
        self.opp = [1, 0] if dims == 1 else [1, 0, 3, 2]
        self.U = nn.Parameter(torch.randn(self.nin, K, generator=g) * init)
        self.Psi = nn.Parameter(torch.randn(self.E, self.nin, self.nin, K, K, generator=g) * init)
        self.O = nn.Parameter(torch.randn(self.nin, K, generator=g) * init)
        self.Q, self.delta, self.infer = Q, delta, infer
        self.w = 1 + self.nb * K

    def q(self, t, hard):
        if not hard:
            return t
        return torch.round(t / self.delta).clamp(-(2 ** (self.Q - 1) - 1), 2 ** (self.Q - 1) - 1) * self.delta

    def tables(self, hard):
        return self.q(self.U, hard), self.q(self.Psi, hard), self.q(self.O, hard)

    # ----- state interface -----
    def init_state(self, xv):
        return xv.new_zeros((self.w,) + tuple(xv.shape[1:]))

    def unpack(self, h):
        B = h.shape[1]
        sp = tuple(h.shape[2:])
        M = h[1:].reshape((self.nb, self.K, B) + sp)
        return M.permute((0, 2) + tuple(range(3, 3 + len(sp))) + (1,))

    def pack(self, y, M):
        sp = tuple(M.shape[2:-1])
        Mc = M.permute((0, M.dim() - 1, 1) + tuple(range(2, 2 + len(sp)))).reshape((self.nb * self.K, M.shape[1]) + sp)
        return torch.cat([y.unsqueeze(0), Mc], 0)

    def step(self, h, xv, mode="soft"):
        hard = mode == "hard"
        xi = xidx(xv)
        Mn = self.messages(self.unpack(h), xi, hard)
        y, _ = self.belief(Mn, xi, hard)
        return self.pack(y, Mn)

    # ----- core -----
    def messages(self, M, xi, hard):
        U, Psi, O = self.tables(hard)
        pad = (0, 0) + (1, 1) * self.dims
        Mp = F.pad(M, pad)  # virtual ring cells receive nothing (uniform)
        xp = F.pad(xi, (1, 1) * self.dims)  # virtual ring cells have x~ = 0
        tot = U[xp] + Mp.sum(0)
        op = torch.logsumexp if (self.infer == "sum") else (lambda t, dim: t.max(dim)[0])
        out = []
        if self.dims == 1:
            N = M.shape[2]
            for d in range(2):
                s = tot - Mp[self.opp[d]]
                if d == 0:
                    sj, xj, xi_ = s[:, 0:N], xp[:, 0:N], xp[:, 1:N + 1]
                    m = op(Psi[0][xj, xi_] + sj.unsqueeze(-1), -2)
                else:
                    sj, xj, xi_ = s[:, 2:N + 2], xp[:, 2:N + 2], xp[:, 1:N + 1]
                    m = op(Psi[0][xi_, xj] + sj.unsqueeze(-2), -1)
                out.append(m)
        else:
            H, W = M.shape[2], M.shape[3]
            for d in range(4):
                s = tot - Mp[self.opp[d]]
                if d == 0:
                    sj, xj, xi_ = s[:, 0:H, 1:W + 1], xp[:, 0:H, 1:W + 1], xp[:, 1:H + 1, 1:W + 1]
                    m = op(Psi[0][xj, xi_] + sj.unsqueeze(-1), -2)
                elif d == 1:
                    sj, xj, xi_ = s[:, 2:H + 2, 1:W + 1], xp[:, 2:H + 2, 1:W + 1], xp[:, 1:H + 1, 1:W + 1]
                    m = op(Psi[0][xi_, xj] + sj.unsqueeze(-2), -1)
                elif d == 2:
                    sj, xj, xi_ = s[:, 1:H + 1, 0:W], xp[:, 1:H + 1, 0:W], xp[:, 1:H + 1, 1:W + 1]
                    m = op(Psi[1][xj, xi_] + sj.unsqueeze(-1), -2)
                else:
                    sj, xj, xi_ = s[:, 1:H + 1, 2:W + 2], xp[:, 1:H + 1, 2:W + 2], xp[:, 1:H + 1, 1:W + 1]
                    m = op(Psi[1][xi_, xj] + sj.unsqueeze(-2), -1)
                out.append(m)
        Mn = torch.stack(out, 0)
        Mn = Mn - Mn.mean(-1, keepdim=True)
        if hard:
            Mn = torch.round(Mn / self.delta).clamp(-(2 ** (self.Q - 1) - 1), 2 ** (self.Q - 1) - 1) * self.delta
        return Mn

    def belief(self, M, xi, hard):
        U, Psi, O = self.tables(hard)
        b = U[xi] + M.sum(0)
        if hard:
            hs = b.argmax(-1)  # ties -> lowest index
            return (O[xi].gather(-1, hs.unsqueeze(-1)).squeeze(-1) > 0).float(), b
        return (torch.softmax(b, -1) * torch.sigmoid(O[xi])).sum(-1), b

    def forward(self, xv, T, hard=False):
        xi = xidx(xv)
        M = xv.new_zeros((self.nb, xv.shape[1]) + tuple(xv.shape[2:]) + (self.K,))
        for _ in range(T):
            M = self.messages(M, xi, hard)
        y, b = self.belief(M, xi, hard)
        return y, M, b

    def gates_per_position(self):
        return 0

    def regs_per_position(self):
        return self.nb * self.K * self.Q + 1


# ====================================================================================== ternary arc consistency
class TernaryNet(nn.Module):
    """Per cell a set of still-possible values stored as V possibility bits; (1,1) = unknown, (0,0) = contradiction.
    Update (synchronous arc consistency, monotone in the information order; Kleene iteration from all-unknown
    converges to the least fixed point): c_h(i)' = u[x~_i](h) AND AND_d OR_h' ( r(h',h) AND c_h'(j_d) ).
    Readout o[x~](S) is indexed by the whole set S. State seen by the evaluators: [y, c_0, ..., c_{V-1}]."""

    def __init__(self, dims, c, V=2):
        super().__init__()
        self.dims, self.c, self.V = dims, c, V
        self.nin = 1 << (c + 1)
        self.E = dims
        self.nb = 2 if dims == 1 else 4
        self.nS = 1 << V
        self.u = nn.Parameter(torch.ones(self.nin, V), requires_grad=False)
        self.r = nn.Parameter(torch.ones(self.E, self.nin, self.nin, V, V), requires_grad=False)
        self.o = nn.Parameter(torch.zeros(self.nin, self.nS), requires_grad=False)
        self.w = 1 + V

    def init_state(self, xv):
        h = xv.new_ones((self.w,) + tuple(xv.shape[1:]))
        h[0] = 0.0
        return h

    def step(self, h, xv, mode="hard"):
        xi = xidx(xv)
        sp_n = h.dim() - 2
        c = h[1:].permute((1,) + tuple(range(2, 2 + sp_n)) + (0,))
        cn = self.update(c, xi)
        y = self.readout(cn, xi)
        cc = cn.permute((cn.dim() - 1, 0) + tuple(range(1, 1 + sp_n)))
        return torch.cat([y.unsqueeze(0), cc], 0)

    def update(self, c, xi):
        r = self.r
        pad = (0, 0) + (1, 1) * self.dims
        cp = F.pad(c, pad, value=1.0)  # virtual ring cells: unknown
        xp = F.pad(xi, (1, 1) * self.dims)
        if self.dims == 1:
            N = c.shape[1]
            xi_ = xp[:, 1:N + 1]
            nbrs = [(cp[:, 0:N], r[0][xp[:, 0:N], xi_]),
                    (cp[:, 2:N + 2], r[0][xi_, xp[:, 2:N + 2]].transpose(-1, -2))]
        else:
            H, W = c.shape[1], c.shape[2]
            xi_ = xp[:, 1:H + 1, 1:W + 1]
            nbrs = [(cp[:, 0:H, 1:W + 1], r[0][xp[:, 0:H, 1:W + 1], xi_]),
                    (cp[:, 2:H + 2, 1:W + 1], r[0][xi_, xp[:, 2:H + 2, 1:W + 1]].transpose(-1, -2)),
                    (cp[:, 1:H + 1, 0:W], r[1][xp[:, 1:H + 1, 0:W], xi_]),
                    (cp[:, 1:H + 1, 2:W + 2], r[1][xi_, xp[:, 1:H + 1, 2:W + 2]].transpose(-1, -2))]
        out = self.u[xi_]
        for cj, R in nbrs:
            out = out * (R * cj.unsqueeze(-1)).amax(dim=-2)  # on bits: AND over directions, OR over h'
        return out

    def subset_index(self, c):
        return (c * (2 ** torch.arange(self.V, device=c.device)).float()).sum(-1).long()

    def readout(self, c, xi):
        return self.o[xi].gather(-1, self.subset_index(c).unsqueeze(-1)).squeeze(-1)

    def fixed_point(self, xv, Tmax):
        """Iterate until no change (monotone: a step with no change is the fixed point) or Tmax steps."""
        xi = xidx(xv)
        c = xv.new_ones(tuple(xv.shape[1:]) + (self.V,))
        for _ in range(Tmax):
            cn = self.update(c, xi)
            if torch.equal(cn, c):
                break
            c = cn
        return c, xi

    def gates_per_position(self):
        return 0

    def regs_per_position(self):
        return self.V + 1


@torch.no_grad()
def bp_to_ternary(bp, fit_data, thr=THR):
    """Round the learned BP tables to the ternary bit tables (Fable design section 6.7):
      r(ha,hb) = [Psi(ha,hb) - max_h' Psi(ha,h') > -thr] AND [Psi(ha,hb) - max_h' Psi(h',hb) > -thr]
      u(h)     = [U(h) - max_h' U(h') > -thr];     o({h}) = [O(h) > 0]
      o(unknown), o(contradiction): majority label over the masked cells of fit_data that end in that state."""
    assert bp.K == 2
    t = TernaryNet(bp.dims, bp.c, V=2)
    Psi, U, O = bp.Psi.data, bp.U.data, bp.O.data
    rel = Psi - Psi.amax(dim=-1, keepdim=True)
    rel2 = Psi - Psi.amax(dim=-2, keepdim=True)
    t.r.copy_(((rel > -thr) & (rel2 > -thr)).float())
    t.u.copy_(((U - U.amax(dim=-1, keepdim=True)) > -thr).float())
    t.o.zero_()
    t.o[:, 1] = (O[:, 0] > 0).float()  # S = {0}
    t.o[:, 2] = (O[:, 1] > 0).float()  # S = {1}
    xv = to_internal(fit_data["x"])
    y = torch.as_tensor(fit_data["y"]).long()
    mk = torch.as_tensor(fit_data["mask"])
    n = xv.shape[-1]
    c, xi = t.fixed_point(xv, ev.t_max_for("M", n) if bp.dims == 2 else 4 * n)
    s = t.subset_index(c)
    cnt = torch.zeros(t.nin * 4 * 2)
    idx = ((xi * 4 + s) * 2 + y)[mk]
    cnt.index_add_(0, idx, torch.ones(idx.numel()))
    cnt = cnt.view(t.nin, 4, 2)
    for x in range(t.nin):
        for sidx in (0, 3):
            if cnt[x, sidx].sum() > 0:
                t.o[x, sidx] = 1.0 if cnt[x, sidx, 1] > cnt[x, sidx, 0] else 0.0
    t.eval()
    return t


# ====================================================================================== data plumbing
def train_batch(task, rng, B):
    """One training-stream batch (dict with x, y, mask, diff) from the training ranges."""
    if task in A1_TASKS:
        n = tk.train_size(task, rng)
        d = tk.generate(task, rng, B, n)
        d["mask"] = np.ones_like(d["y"], dtype=bool)
        return d
    n, kw = ea.size_kw(task, rng)
    return t2.generate(task, rng, B, n, **kw)


def batch_size(task):
    return 128 if task_info(task)[0] == 1 else 32


def val_specs(task):
    if task in A1_TASKS:
        return [(n,) for n in (SIZES_1D if task_info(task)[0] == 1 else SIZES_2D)]
    return list(ea.VAL_SPECS[task])


def fixed_set(task, spec, count, seed, **kw):
    """Returns (key, n, data). P/A/M: tasks.fixed_set; new tasks: evaluate_a2.fixed_set_a2 (shared with R/F)."""
    key = "_".join(map(str, spec))
    if task in A1_TASKS:
        n = spec[0]
        d = tk.fixed_set(task, n, count, seed, **kw)
        d["mask"] = np.ones_like(d["y"], dtype=bool)
        return key, n, d
    n, d = ea.fixed_set_a2(task, spec, count, seed)
    return key, n, d


def n_val(task):
    return tk.N_VAL[task] if task in A1_TASKS else ea.N_VAL


def run_eval(model, task, d, n, mode="hard"):
    """P/A/M -> sim/evaluate.run_hard; new tasks -> sim/evaluate_a2.run_masked (halting + scoring on the mask)."""
    if task in A1_TASKS:
        return ev.run_hard(model, "R", task, d, n, None, m=1, mode=mode)
    return ea.run_masked(model, "R", 2, d, n, None, m=1, mode=mode)


def score(model, task, seed=VAL_SEED, mode="hard", count=None, specs=None):
    """Selection score (section 3.5): mean 'halted and correct' over the in-distribution specs."""
    per = {}
    for spec in (specs or val_specs(task)):
        key, n, d = fixed_set(task, spec, count or n_val(task), seed)
        r = run_eval(model, task, d, n, mode)
        per[key] = ev.summarize(r, d["diff"])
        per[key]["t_fp"] = per[key]["mean_t"]
        if task == "inpaint" and "edge" in d:
            e = d["edge"]
            per[key]["exact_edge"] = float(r["correct"][e].mean()) if e.any() else None
            per[key]["exact_noedge"] = float(r["correct"][~e].mean()) if (~e).any() else None
            per[key]["bitacc_edge"] = float(r["bitacc"][e].mean()) if e.any() else None
            per[key]["edge_frac"] = float(e.mean())
    return float(np.mean([v["exact"] for v in per.values()])), per


# ====================================================================================== deployment / diagnostics
def deploy_model(bp, task, fit_data=None, ternary=None):
    if DEPLOY[task] == "ternary":
        return ternary if ternary is not None else bp_to_ternary(bp, fit_data)
    m = copy.deepcopy(bp)
    m.infer = "max"
    m.eval()
    return m


@torch.no_grad()
def cell_states(model, xv, Tmax):
    """Four-state classification of every cell at the deployed fixed point: index c0 + 2*c1
    (0 contradiction, 1 {0}, 2 {1}, 3 unknown). For the LLR-BP deployment the state is the sign of the belief LLR
    (0 -> unknown; contradiction cannot occur). Returns (state (B,*sp) long, prediction (B,*sp) bool)."""
    if isinstance(model, TernaryNet):
        c, xi = model.fixed_point(xv, Tmax)
        return model.subset_index(c), model.readout(c, xi) > 0.5
    xi = xidx(xv)
    M = xv.new_zeros((model.nb, xv.shape[1]) + tuple(xv.shape[2:]) + (model.K,))
    for _ in range(Tmax):
        Mn = model.messages(M, xi, True)
        if torch.equal(Mn, M):
            break
        M = Mn
    y, b = model.belief(M, xi, True)
    llr = b[..., 1] - b[..., 0]
    s = torch.where(llr > 0, 2, torch.where(llr < 0, 1, 3))
    return s, y > 0.5


@torch.no_grad()
def four_state_counts(model, task, d, n):
    """Counts of (state, label) over masked cells, plus the confusion of predictions within each state."""
    dims = task_info(task)[0]
    Tmax = ev.t_max_for(task, n) if task in A1_TASKS else ea.t_max_dims(dims, n)
    xv = to_internal(d["x"])
    s, pred = cell_states(model, xv, Tmax)
    y = torch.as_tensor(d["y"]) > 0.5
    mk = torch.as_tensor(d["mask"])
    counts = [[int(((s == k) & (y == lab) & mk).sum()) for lab in (0, 1)] for k in range(4)]
    pred1 = [[int(((s == k) & (y == lab) & mk & pred).sum()) for lab in (0, 1)] for k in range(4)]
    tot = max(1, int(mk.sum()))
    return dict(states=list(STATE_NAMES), counts_by_label=counts, predicted_one_by_label=pred1,
                frac=[sum(c) / tot for c in counts], n_cells=tot)


def tail_lengths(dirs, on_cycle):
    """Per cell: number of arrow steps to the first on-cycle cell (0 on a cycle)."""
    n = dirs.shape[0]
    tx = np.zeros_like(dirs); ty = np.zeros_like(dirs)
    for k, (dx, dy) in enumerate(t2.DIRS):
        sel = dirs == k
        yy, xx = np.nonzero(sel)
        tx[sel], ty[sel] = xx + dx, yy + dy
    tail = np.where(on_cycle, 0, -1)
    for _ in range(n * n):
        new = (tail < 0) & (tail[ty, tx] >= 0)
        if not new.any():
            break
        tail[new] = tail[ty, tx][new] + 1
    return tail


@torch.no_grad()
def small_graph_diag(model, task, seed=VAL_SEED, count=100, sizes=None):
    """Section 5: compare the deployed fixed point with the reference labels (cycle_labels / reach_source_labels,
    which the generator uses) graph by graph on small grids; FP / FN / unknown-or-contradiction counts by tail
    length. In-distribution only (n <= 6). cycles_reach needs >= 2 loops, impossible on 4 x 4, so it uses {5, 6}."""
    assert task in ("cycles", "cycles_reach")
    sizes = sizes or (5, 6)  # Q and Q+ both need >= 2 non-adjacent loops, impossible on 4 x 4
    out = {}
    for n in sizes:
        rng = np.random.default_rng([seed, n, 99])
        d = t2.generate(task, rng, count, n)
        xv = to_internal(d["x"])
        s, pred = cell_states(model, xv, n * n)
        s, pred = s.numpy(), pred.numpy()
        y = d["y"] > 0.5
        dirs = (d["x"][:, 0] > 0.5).astype(np.int64) * 2 + (d["x"][:, 1] > 0.5).astype(np.int64)
        on = np.stack([t2.cycle_labels(dirs[k])[0] for k in range(count)])
        tails = np.stack([tail_lengths(dirs[k], on[k]) for k in range(count)])
        by_tail = {}
        for tl in sorted(np.unique(tails).tolist()):
            sel = tails == tl
            by_tail[str(tl)] = dict(cells=int(sel.sum()), fp=int((sel & pred & ~y).sum()), fn=int((sel & ~pred & y).sum()),
                                    unknown=int((sel & (s == 3)).sum()), contradiction=int((sel & (s == 0)).sum()))
        ok = (pred == y).reshape(count, -1).all(1)
        out[str(n)] = dict(graphs=count, graphs_exact=float(ok.mean()), cells=int(y.size), fp=int((pred & ~y).sum()),
                           fn=int((~pred & y).sum()), unknown=int((s == 3).sum()), contradiction=int((s == 0).sum()),
                           by_tail=by_tail)
    return out


def m4_lower_bound(task, deploy):
    """Formula lower bound (brief_A2 section 5: 'lower bound' only; no synthesis in this stage)."""
    dims, c = task_info(task)
    nb = 2 if dims == 1 else 4
    if deploy == "ternary":
        rom = max(1, (1 << (2 * (c + 1))) // 64)  # LUT6 per ROM output bit (mux overhead not counted)
        lut = nb * (4 * rom + 2) + 1 + 2 + 1 + 1
        ff = 2 + 1 + (c + 1)
        return dict(G=0, effective=0, regs=3, lut6_lower_bound=lut, ff_lower_bound=ff, deploy=deploy,
                    note="formula lower bound (Fable design 6.3); measured numbers reported separately")
    Q = Q_BITS
    lut = nb * (3 * Q + 53) + nb * 7 * 2 + 3 + 3  # Fable design 1.3 extrapolated to Q bits, fully parallel
    ff = nb * 2 * Q + 1 + (c + 1)
    return dict(G=0, effective=0, regs=nb * 2 * Q + 1, lut6_lower_bound=lut, ff_lower_bound=ff, deploy=deploy,
                note="formula lower bound (Fable design 1.3, Q=6, fully parallel); measured numbers reported separately")


# ====================================================================================== oracles (tests)
def _bit(x, k):
    return (x >> k) & 1


def plant_bp_oracle(task, lam=4.0, infer="max"):
    """Hand-built K=2 LLR tables (Fable design 1.6) = reviews/a2/fable/oracle.py."""
    dims, c = task_info(task)
    m = BPNet(dims, c, K=2, infer=infer)
    nin = m.nin
    with torch.no_grad():
        m.U.zero_(); m.Psi.zero_(); m.O.zero_()
        if task == "P":
            m.U[0] = torch.tensor([lam, -lam])
            for xa in range(nin):
                for xb in range(nin):
                    if _bit(xb, 1) == 0:
                        continue
                    for ha in range(2):
                        for hb in range(2):
                            m.Psi[0, xa, xb, ha, hb] = lam if hb == (ha ^ _bit(xb, 0)) else -lam
            for x in range(nin):
                m.O[x] = torch.tensor([-lam, lam])
        elif task == "A":
            for xa in range(nin):
                for xb in range(nin):
                    if _bit(xb, 2) == 0:
                        continue
                    aa, bb = _bit(xa, 0), _bit(xa, 1)
                    for ca in range(2):
                        cb_true = 1 if aa + bb + ca >= 2 else 0
                        for cb in range(2):
                            m.Psi[0, xa, xb, ca, cb] = lam if cb == cb_true else -lam
            for x in range(nin):
                for cc in range(2):
                    m.O[x, cc] = lam if (_bit(x, 0) ^ _bit(x, 1) ^ cc) else -lam
        elif task == "M":
            for x in range(nin):
                op, sr, va = _bit(x, 0), _bit(x, 1), _bit(x, 2)
                if va and op and sr:
                    m.U[x] = torch.tensor([-lam, lam])
                elif va and not op:
                    m.U[x] = torch.tensor([lam, -lam])
                m.O[x] = torch.tensor([-lam, lam])
            for e in range(2):
                for xa in range(nin):
                    for xb in range(nin):
                        if _bit(xa, 0) and _bit(xb, 0) and _bit(xa, 2) and _bit(xb, 2):
                            m.Psi[e, xa, xb, 0, 1] = -lam
                            m.Psi[e, xa, xb, 1, 0] = -lam
        else:
            raise ValueError(task)
    m.eval()
    return m


def plant_ternary_oracle(task):
    """Hand-built ternary bit tables (Fable design 6.5) = reviews/a2/fable/ternary.py --oracle."""
    dims, c = task_info(task)
    t = TernaryNet(dims, c, V=2)
    nin = t.nin
    with torch.no_grad():
        t.u.fill_(1.0); t.r.fill_(1.0); t.o.zero_()
        if task == "P":
            for xa in range(nin):
                for xb in range(nin):
                    if _bit(xb, 1) == 0:
                        continue
                    for ha in range(2):
                        for hb in range(2):
                            t.r[0, xa, xb, ha, hb] = 1.0 if hb == (ha ^ _bit(xb, 0)) else 0.0
            t.r[0, 0, :, :, :] = 0.0
            for xb in range(nin):
                if _bit(xb, 1):
                    t.r[0, 0, xb, :, _bit(xb, 0)] = 1.0
            t.o[:, 2] = 1.0
        elif task == "A":
            for xa in range(nin):
                for xb in range(nin):
                    if _bit(xb, 2) == 0:
                        continue
                    aa, bb = _bit(xa, 0), _bit(xa, 1)
                    for ca in range(2):
                        cb_true = 1 if aa + bb + ca >= 2 else 0
                        for cb in range(2):
                            t.r[0, xa, xb, ca, cb] = 1.0 if cb == cb_true else 0.0
            for x in range(nin):
                for s in (1, 2):
                    cc = 1 if s == 2 else 0
                    t.o[x, s] = 1.0 if (_bit(x, 0) ^ _bit(x, 1) ^ cc) else 0.0
        elif task == "M":
            for x in range(nin):
                op, sr, va = _bit(x, 0), _bit(x, 1), _bit(x, 2)
                if va and op and sr:
                    t.u[x] = torch.tensor([0.0, 1.0])
                elif va and not op:
                    t.u[x] = torch.tensor([1.0, 0.0])
                t.o[x, 2] = 1.0
            for e in range(2):
                for xa in range(nin):
                    for xb in range(nin):
                        if _bit(xa, 0) and _bit(xb, 0) and _bit(xa, 2) and _bit(xb, 2):
                            t.r[e, xa, xb, 0, 1] = 0.0
                            t.r[e, xa, xb, 1, 0] = 0.0
        else:
            raise ValueError(task)
    t.eval()
    return t


# ====================================================================================== provenance
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def provenance():
    files = ["tasks.py", "tasks_a2.py", "evaluate.py", "evaluate_a2.py", "b_arm.py"]
    return dict(git_commit=git_commit(), sha256={f: sha256(os.path.join(HERE, f)) for f in files})


# ====================================================================================== training
def masked_bce(p, y, mask):
    p = (1e-5 + (1 - 2e-5) * p).clamp(1e-5, 1 - 1e-5)
    w = mask.float()
    return (F.binary_cross_entropy(p, y, reduction="none") * w).sum() / w.sum().clamp(min=1)


def fit_batch(task, seed, step):
    """The ONE training-stream batch used for o(unknown)/o(contradiction) (section 3.3): training ranges and
    generator, drawn from a stream seeded by (seed, step) so that every checkpoint is reproducible."""
    rng = np.random.default_rng([seed, 9000 + step])
    return train_batch(task, rng, batch_size(task))


def save_ckpt(path, bp, step, ternary, fit_step):
    torch.save(dict(bp=bp.state_dict(), step=step, fit_step=fit_step,
                    ternary=(ternary.state_dict() if ternary is not None else None)), path)


def load_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    task = cfg["task"]
    dims, c = task_info(task)
    out = {}
    for name in ("best_deploy", "best_soft", "last"):
        ck = torch.load(os.path.join(run_dir, name + ".pt"), map_location="cpu")
        bp = BPNet(dims, c, K=2, Q=cfg["Q"], delta=cfg["delta"], infer="sum")
        bp.load_state_dict(ck["bp"])
        bp.eval()
        tern = None
        if ck.get("ternary") is not None:
            tern = TernaryNet(dims, c, V=2)
            tern.load_state_dict(ck["ternary"])
            tern.eval()
        out[name] = dict(bp=bp, ternary=tern, step=ck["step"], deploy=deploy_model(bp, task, ternary=tern) if DEPLOY[task] == "ternary" else deploy_model(bp, task))
    return cfg, out


def train(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    task = a.task
    dims, c = task_info(task)
    model = BPNet(dims, c, K=2, Q=Q_BITS, delta=DELTA, infer="sum", seed=a.seed)
    rng = np.random.default_rng(a.seed + 1000)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    B = batch_size(task)
    cfg = dict(arm="B", task=task, seed=a.seed, K=2, Q=Q_BITS, delta=DELTA, lr=LR, beta=BETA, batch=B, steps=a.steps,
               eval_every=a.eval_every, deploy=DEPLOY[task], thr=THR, n_val=a.n_val, threads=a.threads, out=a.out,
               val_specs=val_specs(task), readout=1, **provenance())
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log, t0 = [], time.time()
    best = {"best_deploy": (-1.0, -1), "best_soft": (-1.0, -1)}
    for step in range(1, a.steps + 1):
        d = train_batch(task, rng, B)
        xv = to_internal(d["x"])
        y = torch.as_tensor(d["y"])
        mk = torch.as_tensor(d["mask"])
        T = max(1, int(math.ceil(float(rng.uniform(1.0, BETA)) * int(d["diff"].max()))))
        p, _, _ = model(xv, T)
        loss = masked_bce(p, y, mk)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                tern = bp_to_ternary(model, fit_batch(task, a.seed, step)) if DEPLOY[task] == "ternary" else None
                dep = deploy_model(model, task, ternary=tern)
                sc_d, per_d = score(dep, task, VAL_SEED, "hard", a.n_val)
                sc_s, per_s = score(model, task, VAL_SEED, "soft", a.n_val)
            model.train()
            rec = dict(step=step, loss=float(loss.detach()), T=T, sec=round(time.time() - t0, 1), deploy_score=sc_d,
                       soft_score=sc_s, deploy_per=per_d, soft_per=per_s)
            log.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_per")}), flush=True)
            if sc_d > best["best_deploy"][0]:  # ties -> earlier step
                best["best_deploy"] = (sc_d, step)
                save_ckpt(os.path.join(a.out, "best_deploy.pt"), model, step, tern, step)
            if sc_s > best["best_soft"][0]:
                best["best_soft"] = (sc_s, step)
                save_ckpt(os.path.join(a.out, "best_soft.pt"), model, step, tern, step)
            if step == a.steps:
                save_ckpt(os.path.join(a.out, "last.pt"), model, step, tern, step)
            json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    return finalize(a.out, a.n_val)


def finalize(run_dir, count=None):
    """final.json: val (777) and val2 (778) for the three checkpoints; 'val'/'val2' at top level = best_deploy
    (deployed), for report.py compatibility; four-state counts on val for best_deploy."""
    cfg, ck = load_run(run_dir)
    task = cfg["task"]
    res = dict(task=task, seed=cfg["seed"], deploy=cfg["deploy"], thr=cfg["thr"],
               best_deploy_step=ck["best_deploy"]["step"], best_soft_step=ck["best_soft"]["step"], last_step=ck["last"]["step"],
               checkpoints={})
    with torch.no_grad():
        for name in ("best_deploy", "best_soft", "last"):
            entry = {}
            for vname, seed in (("val", VAL_SEED), ("val2", VAL2_SEED)):
                sc, per = score(ck[name]["deploy"], task, seed, "hard", count)
                entry[vname] = dict(score=sc, per_size=per)
            if name == "best_soft":
                sc, per = score(ck[name]["bp"], task, VAL_SEED, "soft", count)
                entry["val_soft_mode"] = dict(score=sc, per_size=per)
            res["checkpoints"][name] = entry
        res["val"] = res["checkpoints"]["best_deploy"]["val"]
        res["val2"] = res["checkpoints"]["best_deploy"]["val2"]
        spec = val_specs(task)[-1]
        key, n, d = fixed_set(task, spec, count or n_val(task), VAL_SEED)
        res["four_state_val"] = dict(spec=key, **four_state_counts(ck["best_deploy"]["deploy"], task, d, n))
        if task in ("cycles", "cycles_reach"):
            res["small_graph_diag"] = small_graph_diag(ck["best_deploy"]["deploy"], task, VAL_SEED, 100)
        res["M4"] = m4_lower_bound(task, cfg["deploy"])
    json.dump(res, open(os.path.join(run_dir, "final.json"), "w"), indent=1)
    print("FINAL", json.dumps(dict(task=task, seed=cfg["seed"], best_deploy_step=res["best_deploy_step"],
                                   val=res["val"]["score"], val2=res["val2"]["score"],
                                   best_soft_val=res["checkpoints"]["best_soft"]["val"]["score"],
                                   last_val=res["checkpoints"]["last"]["val"]["score"])), flush=True)
    return res


# ====================================================================================== test-set evaluation
def test_specs(task, locked):
    """Locked: in-distribution + extrapolation sizes with the test seeds (P/A/M: tasks.TEST_SEED = 12345;
    new tasks: evaluate_a2.TEST_SEED_A2 = 24680). Hook: validation seed 777, in-distribution size only."""
    if task in A1_TASKS:
        info = tk.TASKS[task]
        ind = (info["indist"],)
        extra = [(n,) for n in info["extra"]]
        bar = (info["bar"],)
        seed, count = (tk.TEST_SEED, tk.N_TEST[task]) if locked else (VAL_SEED, tk.N_VAL[task])
    else:
        info = t2.TASKS_A2[task]
        ind = tuple(info["indist"])
        extra = [tuple(s) for s in info["extra"]]
        bar = {"cycles": (32,), "cycles_reach": (32,), "inpaint": (32, 16), "life": (64,)}[task]
        seed, count = (ea.TEST_SEED_A2, ea.N_TEST) if locked else (VAL_SEED, ea.N_VAL)
    if not locked:
        extra, bar = [ind], ind  # hook: never leave the in-distribution size
    return ind, extra, bar, seed, count


def eval_test(cfg, ck, locked):
    """Same fields as sim/evaluate.eval_test / evaluate_a2.eval_test_a2 for best_deploy, plus best_soft_M2, last_M2,
    four-state confusion counts and t_fp. The hook (locked=False) runs this code on validation-seed data."""
    from scipy.stats import spearmanr
    task = cfg["task"]
    ind, extra, bar, seed, count = test_specs(task, locked)
    specs = [ind] + [s for s in extra if s != ind]
    dep = ck["best_deploy"]["deploy"]
    per, pool_th, pool_h, pool_d, pool_ok = {}, [], [], [], []
    for spec in specs:
        kw = {"fullchain_p": 0.0} if task == "A" else {}
        key, n, d = fixed_set(task, spec, count, seed, **kw)
        t0 = time.time()
        r = run_eval(dep, task, d, n, "hard")
        per[key] = ev.summarize(r, d["diff"])
        per[key]["t_fp"] = per[key]["mean_t"]
        per[key]["sec"] = round(time.time() - t0, 1)
        if task == "inpaint" and "edge" in d:
            e = d["edge"]
            per[key]["exact_edge"] = float(r["correct"][e].mean()) if e.any() else None
            per[key]["exact_noedge"] = float(r["correct"][~e].mean()) if (~e).any() else None
            per[key]["edge_frac"] = float(e.mean())
        if task == "A":
            _, _, ds = fixed_set(task, spec, count, seed, force_fullchain=True)
            rs = run_eval(dep, task, ds, n, "hard")
            per[key]["stress"] = ev.summarize(rs, ds["diff"])
            bins = (("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16), ("17-32", 17, 32), ("33+", 33, 10 ** 9))
            per[key]["bins_stress"] = {k: (float(rs["correct"][(ds["diff"] >= lo) & (ds["diff"] <= hi)].mean()) if ((ds["diff"] >= lo) & (ds["diff"] <= hi)).any() else None) for k, lo, hi in bins}
            per[key]["bins_uniform"] = {k: (float(r["correct"][(d["diff"] >= lo) & (d["diff"] <= hi)].mean()) if ((d["diff"] >= lo) & (d["diff"] <= hi)).any() else None) for k, lo, hi in bins}
        pool_th.append(r["t_halt"]); pool_h.append(r["halted"]); pool_d.append(d["diff"]); pool_ok.append(r["correct"])
        print(f"  {key}: exact {per[key]['exact']:.3f} halt {per[key]['halt_rate']:.3f} t_halt {per[key]['mean_t_halt']:.1f} t_fp {per[key]['t_fp']:.1f}", flush=True)
    kind = "_".join(map(str, ind))
    kex = ["_".join(map(str, s)) for s in extra]
    kbar = "_".join(map(str, bar))
    out = dict(per_size=per, M1=per[kind]["exact"], M2=float(np.mean([per[k]["exact"] for k in kex])), bar=per[kbar]["exact"],
               hook=not locked, test_seed=seed, n_test=count)
    for suf, key in (("_v2", "exact_v2"), ("_ws", "exact_ws")):
        out["M1" + suf] = per[kind].get(key, per[kind]["exact"])
        out["M2" + suf] = float(np.mean([per[k].get(key, per[k]["exact"]) for k in kex]))
        out["bar" + suf] = per[kbar].get(key, per[kbar]["exact"])
    for w in ("w8", "wT8", "wT4", "wT2", "wTmax"):
        out["M1_" + w] = per[kind].get("exact_" + w, per[kind]["exact"])
        out["M2_" + w] = float(np.mean([per[k].get("exact_" + w, per[k]["exact"]) for k in kex]))
        out["bar_" + w] = per[kbar].get("exact_" + w, per[kbar]["exact"])
    out["outcome_frac_extra"] = [float(np.mean([per[k].get("outcome_frac", [0, 0, 0])[i] for k in kex])) for i in range(3)]
    th, hh, dd, ok = (np.concatenate(p) for p in (pool_th, pool_h, pool_d, pool_ok))
    out["rho_halt"] = float(spearmanr(th[hh], dd[hh])[0]) if hh.sum() > 2 and np.unique(dd[hh]).size > 1 else None
    out["rho_halt_correct"] = float(spearmanr(th[hh & ok], dd[hh & ok])[0]) if (hh & ok).sum() > 2 and np.unique(dd[hh & ok]).size > 1 else None
    out["t_fp"] = {k: per[k]["t_fp"] for k in per}
    out["t_fp_extra"] = float(np.mean([per[k]["t_fp"] for k in kex]))
    # the other two checkpoints (deployed the same way; best_soft also in soft mode), extrapolation sizes only
    for name in ("best_soft", "last"):
        ex = []
        for spec in extra:
            key, n, d = fixed_set(task, spec, count, seed, **({"fullchain_p": 0.0} if task == "A" else {}))
            ex.append(ev.summarize(run_eval(ck[name]["deploy"], task, d, n, "hard"), d["diff"])["exact"])
        out[name + "_M2"] = float(np.mean(ex))
    ex = []
    for spec in extra:
        key, n, d = fixed_set(task, spec, count, seed, **({"fullchain_p": 0.0} if task == "A" else {}))
        ex.append(ev.summarize(run_eval(ck["best_soft"]["bp"], task, d, n, "soft"), d["diff"])["exact"])
    out["best_soft_softmode_M2"] = float(np.mean(ex))
    # four-state confusion on the in-distribution set and on the bar size
    fs = {}
    for spec in {ind, bar}:
        key, n, d = fixed_set(task, spec, count, seed, **({"fullchain_p": 0.0} if task == "A" else {}))  # same data as the main score
        fs[key] = four_state_counts(dep, task, d, n)
    out["four_state"] = fs
    out["M4"] = m4_lower_bound(task, cfg["deploy"])
    out["deploy"] = cfg["deploy"]
    out["checkpoint_steps"] = {k: ck[k]["step"] for k in ck}
    return out


def evaluate(a):
    torch.set_num_threads(a.threads)
    cfg, ck = load_run(a.run_dir)
    if a.locked:
        print("LOCKED evaluation: test seeds (12345 / 24680) and extrapolation sizes", flush=True)
        res = eval_test(cfg, ck, True)
        json.dump(res, open(os.path.join(a.run_dir, "test.json"), "w"), indent=1)
    elif a.hook:
        res = eval_test(cfg, ck, False)
        json.dump(res, open(os.path.join(a.run_dir, "test_hook.json"), "w"), indent=1)
    else:
        with torch.no_grad():
            sc, per = score(ck["best_deploy"]["deploy"], cfg["task"], VAL_SEED, "hard")
        res = dict(score=sc, per_size=per)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_size", "four_state")}, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--task", required=True, choices=list(TASKS))
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--threads", type=int, default=4)
    tr.add_argument("--steps", type=int, default=STEPS, help="(tests only) default 2000 = brief_A2 3.2")
    tr.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    tr.add_argument("--n_val", type=int, default=None, help="(tests only) validation samples per size")
    ev_ = sub.add_parser("evaluate")
    ev_.add_argument("run_dir")
    ev_.add_argument("--locked", action="store_true", help="test seeds + extrapolation sizes -> test.json")
    ev_.add_argument("--hook", action="store_true", help="same code path on val 777, in-distribution only -> test_hook.json")
    ev_.add_argument("--threads", type=int, default=4)
    dg = sub.add_parser("diag")
    dg.add_argument("run_dir")
    dg.add_argument("--count", type=int, default=100)
    dg.add_argument("--threads", type=int, default=4)
    orc = sub.add_parser("oracle")
    orc.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        return train(a)
    if a.cmd == "evaluate":
        return evaluate(a)
    if a.cmd == "diag":
        torch.set_num_threads(a.threads)
        cfg, ck = load_run(a.run_dir)
        res = small_graph_diag(ck["best_deploy"]["deploy"], cfg["task"], VAL_SEED, a.count)
        json.dump(res, open(os.path.join(a.run_dir, "small_graph_diag.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
        return res
    torch.set_num_threads(a.threads)
    out = {}
    with torch.no_grad():
        for task in A1_TASKS:
            for infer in ("max", "sum"):
                sc, per = score(plant_bp_oracle(task, infer=infer), task)
                out[f"{task}/bp_{infer}"] = sc
            sc, per = score(plant_ternary_oracle(task), task)
            out[f"{task}/ternary"] = sc
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
