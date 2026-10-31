"""Chain helpers shared by m1d_cert.py and delay.py (Fable review section 1.2: on a chain the AC fixed point is
S_v = L_v & R_v, with L the left-to-right support scan).  Tables are in sim/a5/ac.py format (1-D: r has shape
(1, nin, nin, K, K)); sets are int bitmasks (bit h = value h is still possible)."""
import numpy as np

from . import ac


def tables_dict(u, r, o):
    return dict(u=np.asarray(u), r=np.asarray(r), o=np.asarray(o))


def u_masks(u):
    K = u.shape[1]
    return (u.astype(np.int64) << np.arange(K)).sum(-1)


def row_masks(r1):
    """r1 = r[0] (nin, nin, K, K) -> rm[xa, xb, g] = bitmask over h of r1[xa, xb, g, h]."""
    K = r1.shape[-1]
    return (r1.astype(np.int64) << np.arange(K)).sum(-1)


def left_scan(u, r, X):
    """X (B, N) int codes (all real).  Returns L (B, N) bitmasks: L_i = U(x_i) & support from L_{i-1}; the cell left of
    cell 0 is the virtual cell (code 0, full domain)."""
    K = u.shape[1]
    um = u_masks(u)
    rm = row_masks(r[0])
    B, N = X.shape
    L = np.zeros((B, N), np.int64)
    prev = np.full(B, (1 << K) - 1, np.int64)
    xprev = np.zeros(B, np.int64)
    for i in range(N):
        x = X[:, i]
        new = np.zeros(B, np.int64)
        for g in range(K):
            new |= np.where((prev >> g) & 1, rm[xprev, x, g], 0)
        new &= um[x]
        L[:, i] = new
        prev, xprev = new, x
    return L


def predict(tables, X, chunk=4096):
    """Fixed-point readout (ac.py) for a batch of 1-D code words X (B, N) -> (B, N) uint8."""
    out = []
    for s in range(0, X.shape[0], chunk):
        xi = X[s:s + chunk]
        c, _ = ac.fixed_point(tables["u"], tables["r"], xi)
        out.append(ac.readout(tables["o"], c, xi))
    return np.concatenate(out, 0)


def ternary_predict(tables, xi):
    """Cross-check with the frozen simulator sim.b_arm.TernaryNet (one word xi (N,) or one grid xi (H, W)).
    Returns the readout array.  Requires torch; sim/ is on sys.path via the sim.a5 package import."""
    import torch
    from b_arm import TernaryNet
    xi = np.asarray(xi)
    u, r, o = tables["u"], tables["r"], tables["o"]
    dims, nin, K = xi.ndim, u.shape[0], u.shape[1]
    c = int(np.log2(nin)) - 1
    m = TernaryNet(dims, c, K)
    for name, arr in (("u", u), ("r", r), ("o", o)):
        getattr(m, name).data.copy_(torch.tensor(arr.astype(np.float32)))
    xt = torch.tensor(xi[None].astype(np.int64))
    cc = torch.ones(tuple(xt.shape) + (K,))
    for _ in range(K * xi.size + 2):
        nxt = m.update(cc, xt)
        if torch.equal(nxt, cc):
            break
        cc = nxt
    return m.readout(cc, xt)[0].numpy().astype(np.int64)
