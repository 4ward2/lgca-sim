"""Hand-built K=16 deployed model for M2D (design_A5.md), in TernaryNet table form.

Latent value h = su + 2*cl + 4*s + 8*c  (bits: copy of up-neighbour sum, copy of left-neighbour carry, own sum, own carry).
Input code = b_j + 2*a_{k-j} + 4*valid.
  u[x](h)  = 1 iff s = su ^ (a&b) ^ cl and c = maj(su, a&b, cl)         (real cells; code 0 row is unused)
  r[0][xa,xb](ha,hb) (vertical, a above b): if b-cell is real:  a virtual -> hb.su == 0 ; a real -> hb.su == ha.s ;
                                            if b-cell is virtual: always 1
  r[1][xa,xb](ha,hb) (horizontal, a left of b): same with cl / c.
  o[x](S) = 1 iff S nonempty and every h in S has s = 1 (the readout needs only the own-sum bit; contradiction -> 0).

BOUNDARY ANCHORING (Fable review section 2.2 -- without it no cell ever loses a value and every readout is wrong):
the virtual out-of-bounds cells have code 0 and a full, never-updated domain.  Through their relation tables they pin
  * top row   (cell above is virtual, xa = 0):  r[0][0, xb](., hb) = [hb.sigma == 0]   (sum input from above is 0)
  * left col  (cell left is virtual, xa = 0):   r[1][0, xb](., hb) = [hb.gamma == 0]   (carry input from left is 0)
  * bottom row / right col (virtual cell is the SECOND cell, xb = 0): relation = 1 (no constraint).
Both builders write these three cases in `_relations` below; `check_anchoring` re-derives them from the finished
tables.

K=4 oracle (Theorem 3, `build4`): latent h = sigma + 2*gamma (copies of the up-neighbour sum and left-neighbour carry);
own s = sigma ^ p ^ gamma and c = maj(sigma, p, gamma) are computed inside the tables from the local code
(p = a & b), never stored in the latent value.  u = all ones, vertical r = [hb.sigma == s(ha, xa)],
horizontal r = [hb.gamma == c(ha, xa)], o[x]({h}) = s(h, x) and 0 on every non-singleton set.
The Kleene iteration reaches the unique fixed point (all cells singleton) in <= 3n - 1 strict-change rounds.
"""
import numpy as np

from . import ac

K = 16
NIN = 8
K4 = 4


def _relations(K, dec, own, r):
    """Fill r (2, NIN, NIN, K, K) for latent decoder dec(h) -> (sigma, gamma) and own(h, x) -> (s, c).
    Case split is explicit:  xb virtual -> free;  xa virtual -> ANCHOR (sigma==0 / gamma==0);  else copy of neighbour."""
    for xa in range(NIN):
        for xb in range(NIN):
            for ha in range(K):
                for hb in range(K):
                    sig_b, gam_b = dec(hb)
                    if not (xb >> 2) & 1:                      # virtual cell below / right of a: no constraint
                        r[0, xa, xb, ha, hb] = 1
                        r[1, xa, xb, ha, hb] = 1
                    elif not (xa >> 2) & 1:                    # virtual cell above / left: boundary anchoring
                        r[0, xa, xb, ha, hb] = int(sig_b == 0)
                        r[1, xa, xb, ha, hb] = int(gam_b == 0)
                    else:                                      # both real: copy neighbour's own sum / carry
                        s_a, c_a = own(ha, xa)
                        r[0, xa, xb, ha, hb] = int(sig_b == s_a)
                        r[1, xa, xb, ha, hb] = int(gam_b == c_a)


def _bit(h, i):
    return (h >> i) & 1


def build():
    u = np.zeros((NIN, K), np.uint8)
    r = np.zeros((2, NIN, NIN, K, K), np.uint8)
    o = np.zeros((NIN, 1 << K), np.uint8)
    for x in range(NIN):
        valid = (x >> 2) & 1
        if not valid:
            u[x, :] = 1  # unused for virtual cells; leave permissive
            continue
        p = (x & 1) & ((x >> 1) & 1)
        for h in range(K):
            su, cl, s, c = (_bit(h, i) for i in range(4))
            m = int(su + p + cl >= 2)
            u[x, h] = int(s == (su ^ p ^ cl) and c == m)
    _relations(K, lambda h: (_bit(h, 0), _bit(h, 1)), lambda h, x: (_bit(h, 2), _bit(h, 3)), r)
    S = np.arange(1 << K, dtype=np.int64)
    ones_ok = np.ones(1 << K, bool)
    for h in range(K):
        if not _bit(h, 2):
            ones_ok &= ((S >> h) & 1) == 0   # members must all have s = 1
    ones_ok &= S != 0
    o[:] = ones_ok.astype(np.uint8)[None, :]
    return dict(u=u, r=r, o=o)


def build4():
    """Theorem 3: the K=4 oracle (see module docstring)."""
    nin = NIN
    u = np.ones((nin, K4), np.uint8)
    r = np.zeros((2, nin, nin, K4, K4), np.uint8)
    o = np.zeros((nin, 1 << K4), np.uint8)

    def dec(h):
        return h & 1, (h >> 1) & 1

    def own(h, x):
        sg, gm = dec(h)
        p = (x & 1) & ((x >> 1) & 1)
        return sg ^ p ^ gm, int(sg + p + gm >= 2)

    _relations(K4, dec, own, r)
    for x in range(nin):
        for h in range(K4):
            o[x, 1 << h] = own(h, x)[0]            # singleton {h}: own sum; every other set stays 0
    return dict(u=u, r=r, o=o)


def build_k(k):
    return {4: build4, 16: build}[k]()


def check_anchoring(T):
    """Re-derive the boundary anchoring from finished K=4 / K=16 tables: for every real b-cell the relation with a
    virtual a-cell (xa = 0, any ha) admits exactly the hb with sigma_b = 0 (vertical) / gamma_b = 0 (horizontal), and
    relations towards a virtual b-cell (xb = 0) are all ones.  Returns dict of booleans."""
    Kk = T["u"].shape[1]
    sig = [(h & 1) for h in range(Kk)]
    gam = [(h >> 1) & 1 for h in range(Kk)]
    ok = dict(top=True, left=True, free_bottom=True, free_right=True)
    for xb in range(NIN):
        if (xb >> 2) & 1:
            for ha in range(Kk):
                for hb in range(Kk):
                    ok["top"] &= int(T["r"][0, 0, xb, ha, hb]) == int(sig[hb] == 0)
                    ok["left"] &= int(T["r"][1, 0, xb, ha, hb]) == int(gam[hb] == 0)
    for xa in range(NIN):
        ok["free_bottom"] &= bool((T["r"][0, xa, 0] == 1).all())
        ok["free_right"] &= bool((T["r"][1, xa, 0] == 1).all())
    return ok


def strict_change_rounds(T, x):
    """Number of strict-change rounds of the Kleene iteration for each grid in x (B, 2, n, 2n)."""
    xi = ac.codes_from_x(x)
    out = []
    for i in range(xi.shape[0]):
        _, steps = ac.fixed_point(T["u"], T["r"], xi[i:i + 1])
        out.append(steps)
    return np.array(out)


def predict(tables, x, tmax=10000):
    """x (B, 2, n, 2n) -> (readout (B, n, 2n), final latent sets c (B, n, 2n, 16), steps)."""
    xi = ac.codes_from_x(x)
    c, steps = ac.fixed_point(tables["u"], tables["r"], xi, tmax)
    return ac.readout(tables["o"], c, xi), c, steps
