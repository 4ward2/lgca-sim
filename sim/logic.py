"""Differentiable logic-gate blocks (difflogic-style) with recurrent / feed-forward wrappers.

Internal layout: features are (D, B, *sp) so gate wiring is an index_select on dim 0.
Each gate output = c0 + c1*a + c2*b + c3*a*b, where (c0..c3) is a softmax mixture over
the 16 two-input Boolean functions. With hard (argmax) coefficients and 0/1 inputs this
is exactly the Boolean gate, i.e. what the exported Verilog would compute.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# rows: op id 0..15; columns: coefficients of [1, a, b, ab]
OPS = torch.tensor([
    [0, 0, 0, 0],    # 0  FALSE
    [0, 0, 0, 1],    # 1  A AND B
    [0, 1, 0, -1],   # 2  A AND NOT B
    [0, 1, 0, 0],    # 3  A
    [0, 0, 1, -1],   # 4  NOT A AND B
    [0, 0, 1, 0],    # 5  B
    [0, 1, 1, -2],   # 6  XOR
    [0, 1, 1, -1],   # 7  OR
    [1, -1, -1, 1],  # 8  NOR
    [1, -1, -1, 2],  # 9  XNOR
    [1, 0, -1, 0],   # 10 NOT B
    [1, 0, -1, 1],   # 11 A OR NOT B
    [1, -1, 0, 0],   # 12 NOT A
    [1, -1, 0, 1],   # 13 NOT A OR B
    [1, 0, 0, -1],   # 14 NAND
    [1, 0, 0, 0],    # 15 TRUE
], dtype=torch.float32)
OP_PASS_A = 3
TRIVIAL_OPS = {0, 3, 5, 10, 12, 15}  # constants, wires, inverters


def make_wiring(in_dim, out_dim, gen):
    """difflogic-style: every source used at least once when 2*out >= in."""
    picks = 2 * out_dim
    if picks >= in_dim:
        idx = torch.randperm(picks, generator=gen) % in_dim
        idx = idx[torch.randperm(picks, generator=gen)]
    else:
        idx = torch.randperm(in_dim, generator=gen)[:picks]
    return idx[:out_dim].clone(), idx[out_dim:].clone()


def make_resid_wiring(in_dim, out_dim, gen, ident, priority=(), xbias=0.0, nbr_src=None):
    """'Residual wiring': input A of gate j is the fixed source ident[j] (identity path),
    input B is random; sources in `priority` (task inputs, valid bits) are each used at least once.
    With xbias > 0, each remaining B input is drawn from `priority` with that probability
    (input recall), otherwise uniformly from all sources."""
    ia = torch.as_tensor(ident, dtype=torch.long)
    pr = torch.as_tensor(list(priority), dtype=torch.long)
    prio_all = pr.clone()
    pr = pr[torch.randperm(len(pr), generator=gen)][:out_dim]
    if nbr_src is not None:
        pr = pr[:0]  # nbr mode: no forced coverage, B is neighbour-channel or (w.p. xbias) a task input
    rest = out_dim - len(pr)
    rnd = torch.randint(0, in_dim, (rest,), generator=gen)
    if nbr_src is not None:
        # same-channel neighbour wiring: B of gate j = channel j of a random neighbour cell
        # (DiffLogic CA pairs each neighbour with the centre cell)
        rnd = torch.tensor([nbr_src[j][torch.randint(0, len(nbr_src[j]), (1,), generator=gen).item()]
                            for j in range(out_dim)][:rest], dtype=torch.long)
    if xbias > 0 and len(prio_all):
        use = torch.rand(rest, generator=gen) < xbias
        rnd[use] = prio_all[torch.randint(0, len(prio_all), (int(use.sum()),), generator=gen)]
    ib = torch.cat([pr, rnd])
    if nbr_src is not None:
        # A is the cell's own state channel; B is a neighbour channel or a task/valid bit: never equal
        assert not (ib == ia).any(), "nbr wiring: B collided with A"
    if nbr_src is None:  # nbr mode keeps gate j aligned with channel j
        ib = ib[torch.randperm(out_dim, generator=gen)]
    # avoid a == b (a gate reading the same wire twice is degenerate)
    clash = ib == ia
    while clash.any():
        ib[clash] = torch.randint(0, in_dim, (int(clash.sum()),), generator=gen)
        clash = ib == ia
    return ia, ib


# truth-table rows (a,b) = 00, 01, 10, 11 -> coefficients of [1, a, b, ab]
IW_BASIS = torch.tensor([[1, -1, -1, 1], [0, 0, 1, -1], [0, 1, 0, -1], [0, 0, 0, 1]], dtype=torch.float32)


class GateLayer(nn.Module):
    """param='softmax': difflogic, softmax over 16 ops.
    param='iw': input-wise, one sigmoid per truth-table row (4 params per gate)."""

    def __init__(self, in_dim, out_dim, gen, resid=0.0, wiring=None, param="softmax", trim=False, ib_cands=None):
        super().__init__()
        # ib_cands: (out_dim, Nc) candidate sources for input B; column 0 is the fixed wiring.
        # Learnable connection (Fully Trainable DLGN, arXiv 2607.09399): forward uses the argmax
        # candidate, backward the softmax mixture (straight-through on connections).
        if ib_cands is not None:
            self.register_buffer("ibc", ib_cands)
            self.cw = nn.Parameter(torch.rand(ib_cands.shape, generator=gen))
        else:
            self.ibc = None
        # trim: forbid the constant gates FALSE/TRUE (Fully Trainable DLGN, arXiv 2607.09399)
        mask = torch.zeros(16)
        if trim:
            mask[[0, 15]] = float("-inf")
        self.register_buffer("mask", mask)
        ia, ib = wiring if wiring is not None else make_wiring(in_dim, out_dim, gen)
        self.register_buffer("ia", ia)
        self.register_buffer("ib", ib)
        self.param = param
        if trim and param == "iw":
            raise ValueError("trim is only implemented for the softmax parametrisation")
        if param == "softmax":
            w = torch.randn(out_dim, 16, generator=gen)
            w[:, OP_PASS_A] += resid
        else:
            w = torch.randn(out_dim, 4, generator=gen)
            w += resid * torch.tensor([-1.0, -1.0, 1.0, 1.0])  # pass-through A
        self.logits = nn.Parameter(w)
        self.register_buffer("ops", OPS.clone())
        self.register_buffer("iwb", IW_BASIS.clone())

    def coefs(self, mode):
        if self.param == "iw":
            hard = (self.logits > 0).float() @ self.iwb
            if mode == "hard":
                return hard
            soft = torch.sigmoid(self.logits) @ self.iwb
        else:
            lg = self.logits + self.mask
            hard = self.ops[lg.argmax(1)]
            if mode == "hard":
                return hard
            soft = F.softmax(lg, 1) @ self.ops
        if mode == "st":  # hard forward, soft backward
            return soft + (hard - soft).detach()
        return soft

    def chosen_ib(self):
        if self.ibc is None:
            return self.ib
        return self.ibc.gather(1, self.cw.argmax(1, keepdim=True)).squeeze(1)

    def forward(self, x, mode="soft"):
        c = self.coefs(mode)
        shape = (-1,) + (1,) * (x.dim() - 1)
        a = x.index_select(0, self.ia)
        if self.ibc is None:
            b = x.index_select(0, self.ib)
        else:
            hard_b = x.index_select(0, self.chosen_ib())
            if mode == "hard":
                b = hard_b
            else:
                out_dim, nc = self.ibc.shape
                cand = x.index_select(0, self.ibc.reshape(-1)).reshape((out_dim, nc) + tuple(x.shape[1:]))
                wsm = F.softmax(self.cw, 1).reshape((out_dim, nc) + (1,) * (x.dim() - 1))
                soft_b = (wsm * cand).sum(1)
                b = soft_b + (hard_b - soft_b).detach()
        c0, c1, c2, c3 = (c[:, k].reshape(shape) for k in range(4))
        return c0 + a * (c1 + c3 * b) + c2 * b

    def op_ids(self):
        if self.param == "iw":
            return ((self.logits > 0).long() * torch.tensor([8, 4, 2, 1])).sum(1)
        return (self.logits + self.mask).argmax(1)


def neighbourhood(z):
    """z: (D, B, N) or (D, B, H, W) -> stacked neighbours (|N|*D, B, ...), zero padded."""
    if z.dim() == 3:
        N = z.shape[-1]
        zp = F.pad(z, (1, 1))
        return torch.cat([zp[..., 0:N], z, zp[..., 2:N + 2]], 0)
    H, W = z.shape[-2:]
    zp = F.pad(z, (1, 1, 1, 1))
    return torch.cat([z,
                      zp[..., 0:H, 1:W + 1], zp[..., 2:H + 2, 1:W + 1],
                      zp[..., 1:H + 1, 0:W], zp[..., 1:H + 1, 2:W + 2]], 0)


class Block(nn.Module):
    """3 gate layers of width w; input = neighbourhood of [state (w_in), task inputs, valid]."""

    def __init__(self, dims, w_in, c, w, gen, resid, layers=3, wiring="resid", param="softmax", xbias=0.0, trim=False, lconn=0):
        # wiring: "resid" (A identity, B random) or "nbr" (A identity, B same channel of a random neighbour)
        # lconn: number of learnable B-input candidates per gate (0 = fixed wiring)
        super().__init__()
        nb = 3 if dims == 1 else 5
        per = w_in + c + 1
        d = nb * per
        # position of the cell itself inside neighbourhood(): 1D [left, self, right], 2D [self, N, S, W, E]
        self_off = per if dims == 1 else 0
        mods = []
        for li in range(layers):
            if wiring in ("resid", "nbr"):
                assert w == w_in
                nbr_src = None
                if li == 0:
                    ident = [self_off + j for j in range(w)]
                    prio = [p * per + w_in + k for p in range(nb) for k in range(c + 1)]
                    others = [p for p in range(nb) if p * per != self_off]
                    nbr_src = [[p * per + j for p in others] for j in range(w)]
                else:
                    ident, prio = list(range(w)), ()
                kw = dict(xbias=xbias if li == 0 else 0.0, nbr_src=nbr_src if (li == 0 and wiring == "nbr") else None)
                wr = make_resid_wiring(d, w, gen, ident, prio, **kw)
                cands = None
                if lconn > 1:
                    extra = [make_resid_wiring(d, w, gen, ident, prio, **kw)[1] for _ in range(lconn - 1)]
                    cands = torch.stack([wr[1]] + extra, 1)
            else:
                wr, cands = None, None
            mods.append(GateLayer(d, w, gen, resid, wr, param, trim, cands))
            d = w
        self.layers = nn.ModuleList(mods)
        self.w = w

    def forward(self, h, xv, mode):
        z = neighbourhood(torch.cat([h, xv], 0))
        for l in self.layers:
            z = l(z, mode)
        return z

    def gate_count(self):
        return sum(l.logits.shape[0] for l in self.layers)


def readout(h, m):
    """Soft: mean of the first m state channels. Hard eval thresholds at > 0.5 (majority vote;
    a tie counts as 0). m=1 is plain channel 0."""
    return h[:m].mean(0)


class LoopNet(nn.Module):
    """arm R: one block iterated with shared weights."""

    def __init__(self, dims, c, G, seed, resid=0.0, param="softmax", xbias=0.0, trim=False, wiring="resid", lconn=0):
        super().__init__()
        assert G % 3 == 0
        gen = torch.Generator().manual_seed(seed)
        self.w = G // 3
        self.block = Block(dims, self.w, c, self.w, gen, resid, param=param, xbias=xbias, trim=trim, wiring=wiring, lconn=lconn)
        self.dims = dims

    def init_state(self, xv):
        return xv.new_zeros((self.w,) + tuple(xv.shape[1:]))

    def step(self, h, xv, mode):
        return self.block(h, xv, mode)

    def forward(self, xv, T, mode="soft", ckpt=False, h=None):
        h = self.init_state(xv) if h is None else h
        for _ in range(T):
            if ckpt and torch.is_grad_enabled():
                h = checkpoint(self.step, h, xv, mode, use_reentrant=False)
            else:
                h = self.step(h, xv, mode)
        return h

    def gates_per_position(self):
        return self.block.gate_count()

    def regs_per_position(self):
        return self.w


class FFNet(nn.Module):
    """arm F (K blocks of width G/(3K)) and arm C (K blocks of width w, untied)."""

    def __init__(self, dims, c, K, w, seed, resid=0.0, param="softmax", xbias=0.0, trim=False, wiring="resid", lconn=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.w = w
        self.blocks = nn.ModuleList([Block(dims, w, c, w, gen, resid, param=param, xbias=xbias, trim=trim, wiring=wiring, lconn=lconn) for _ in range(K)])

    def forward(self, xv, T=None, mode="soft", ckpt=False, h=None):
        h = xv.new_zeros((self.w,) + tuple(xv.shape[1:]))
        for blk in self.blocks:
            if ckpt and torch.is_grad_enabled():
                h = checkpoint(blk, h, xv, mode, use_reentrant=False)
            else:
                h = blk(h, xv, mode)
        return h

    def gates_per_position(self):
        return sum(b.gate_count() for b in self.blocks)

    def regs_per_position(self):
        return 0


def effective_gates(model):
    n = 0
    for m in model.modules():
        if isinstance(m, GateLayer):
            ids = m.op_ids().tolist()
            n += sum(1 for i in ids if i not in TRIVIAL_OPS)
    return n


def to_internal(x_np, device="cpu"):
    """(B, c, *sp) numpy -> (c+1, B, *sp) torch with valid bit appended."""
    x = torch.as_tensor(x_np, device=device).transpose(0, 1).contiguous()
    valid = torch.ones_like(x[:1])
    return torch.cat([x, valid], 0)
