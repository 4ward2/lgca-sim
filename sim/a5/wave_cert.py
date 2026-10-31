"""Causal-wave all-size certificate for M2D tables (Fable review section 2.2, obligations (i)-(iii)), PASS / UNKNOWN / FAIL.

Tables: sim/a5/ac.py format (c = 2: code bit0 = b_j, bit1 = a_{k-j}, bit2 = in-bounds, virtual = 0), any K.

WHAT IS CHECKED (sufficient condition, sound for every grid size n x 2n, n >= 1)
  The "wave" at a real cell is the set obtained by using ONLY the up and left ports:
        W(x; up, left) = U[x]  &  supp_vert(up -> x)  &  supp_horiz(left -> x)
  where the neighbour above / left is either a real cell (its own wave set W') or the virtual cell (code 0, full domain).
  Since the true AC operator is <= the up/left-only operator, the fixed point is contained in the wave sets (computed by
  one sweep in j+k order).  We close the following abstract state space under all triples (code x, up-state, left-state):
        state = (code x, wave set W, true sum s, true carry c),   s' = s_up ^ p ^ c_left,  c' = maj(s_up, p, c_left),
        p = bit0 & bit1 of the cell code, virtual neighbour: s = c = 0 (top-row / left-column anchoring is done by the
        virtual-cell relation tables: they appear here as the support computed from the full virtual domain).
  Neighbouring CODES are treated as arbitrary real codes (over-approximation: the shear consistency is ignored).
  Obligations:
   (i)   every reachable wave set is a SINGLETON {h};
   (ii)  down / right virtual ports: for every reachable state some g exists with r[0][x,0](h,g) = 1 and r[1][x,0](h,g) = 1
         (interior down/right ports are satisfied by construction: each interior edge is the up/left edge of one endpoint,
          and the wave set was defined as supported by it, so the singleton assignment is a consistent solution);
   (iii) readout o[x]({h}) equals the TRUE sum bit s' of the state (dense target: every cell, not just the last row).
  PASS => by induction over j+k: the singleton assignment is an AC post-fixed point, the fixed point is contained in the
  wave sets, hence equals them: every cell of every n x 2n grid reads the true partial-product bit.  Corollary: the Kleene
  iteration is final after round j+k+1 at cell (j,k), so strict-change rounds <= 3n-1 for every n (halting bound).
NOT CHECKED: a table that is correct only because of the shear consistency of neighbouring codes (or correct on the last
  row only) can return UNKNOWN; UNKNOWN is not a claim of incorrectness.
FAIL: some small grid (exhaustive n <= 3, random n <= 6) whose simulated readout (ac.py, re-confirmed with the frozen
  TernaryNet) differs from the target ('all' = dense default, or 'last' = last row only).
"""
import itertools

import numpy as np

from . import ac, chain_utils as cu, tasks_mul as tm

NIN = 8


def _masks(T):
    return dict(um=cu.u_masks(T["u"]), rv=cu.row_masks(T["r"][0]), rh=cu.row_masks(T["r"][1]),
                K=T["u"].shape[1])


def _support(rm, xa, xb, Wa, K):
    """Bitmask of h in cell b supported by the neighbour a (relation first->second) holding set Wa."""
    out = 0
    for g in range(K):
        if (Wa >> g) & 1:
            out |= int(rm[xa, xb, g])
    return out


def _popcount(v):
    return bin(v).count("1")


def wave_check(T, max_states=100000):
    """Returns dict(ok, reason, n_states, n_triples, state (violating triple) ...)."""
    M = _masks(T)
    K = M["K"]
    full = (1 << K) - 1
    um, rv, rh = M["um"], M["rv"], M["rh"]
    real = [x for x in range(NIN) if (x >> 2) & 1]
    V = (0, full, 0, 0)  # virtual neighbour: code 0, full domain, true s = c = 0
    reach = set()
    seen_triples = 0
    changed = True
    while changed:
        changed = False
        states = [V] + sorted(reach)
        for x in real:
            p = (x & 1) & ((x >> 1) & 1)
            for qu, ql in itertools.product(states, states):
                seen_triples += 1
                W = int(um[x]) & _support(rv, qu[0], x, qu[1], K) & _support(rh, ql[0], x, ql[1], K)
                if W == 0 or _popcount(W) != 1:
                    return dict(ok=False, reason="(i) wave set not a singleton", x=x, up=qu, left=ql, W=W,
                                n_states=len(reach), n_triples=seen_triples)
                h = W.bit_length() - 1
                s = qu[2] ^ p ^ ql[3]
                c = int(qu[2] + p + ql[3] >= 2)
                if int(T["o"][x, W]) != s:
                    return dict(ok=False, reason="(iii) readout != true sum bit", x=x, up=qu, left=ql, W=W,
                                n_states=len(reach), n_triples=seen_triples)
                # (ii) virtual down / right ports: exists g with r(h, g) = 1 (virtual cell domain is full)
                if not any(int(T["r"][0, x, 0, h, g]) for g in range(K)) or \
                   not any(int(T["r"][1, x, 0, h, g]) for g in range(K)):
                    return dict(ok=False, reason="(ii) bottom/right virtual port rejects a reachable value", x=x, h=h,
                                n_states=len(reach), n_triples=seen_triples)
                st = (x, W, s, c)
                if st not in reach:
                    reach.add(st)
                    changed = True
                    if len(reach) > max_states:
                        return dict(ok=False, reason="state space too large", n_states=len(reach))
        # loop again until closure
    return dict(ok=True, reason="all obligations hold", n_states=len(reach), n_triples=seen_triples)


def find_counterexample(T, target="all", n_exh=3, n_rand=(4, 5, 6), n_samples=150, seed=200601):
    """Smallest-first search for a grid whose fixed-point readout is wrong. Returns dict or None."""
    def wrong_cells(d):
        y, _, _ = _predict(T, d["x"])
        want = d["y"].astype(np.uint8)
        if target == "last":
            return y[:, -1, :] != want[:, -1, :], y, want
        return y != want, y, want

    for n in range(1, n_exh + 1):
        d = tm.generate_all("M2D", n)
        bad, y, want = wrong_cells(d)
        idx = np.flatnonzero(bad.reshape(len(bad), -1).any(1))
        if len(idx):
            return _cex(T, d, int(idx[0]), n, y, want, target)
    rng = np.random.default_rng(seed)
    for n in n_rand:
        d = tm.generate("M2D", rng, n_samples, n)
        bad, y, want = wrong_cells(d)
        idx = np.flatnonzero(bad.reshape(len(bad), -1).any(1))
        if len(idx):
            return _cex(T, d, int(idx[0]), n, y, want, target)
    return None


def _predict(T, x):
    xi = ac.codes_from_x(x)
    c, steps = ac.fixed_point(T["u"], T["r"], xi)
    return ac.readout(T["o"], c, xi), c, steps


def _cex(T, d, i, n, y, want, target):
    xi = ac.codes_from_x(d["x"][i:i + 1])[0]
    tern = cu.ternary_predict(T, xi)
    return dict(n=n, a=int(d["a"][i]), b=int(d["b"][i]), target=target, pred=y[i].tolist(), want=want[i].tolist(),
                simulator_agrees=bool(np.array_equal(tern, y[i])))


def certify(T, target="all", search=True):
    """PASS (all-size theorem for the dense target, hence also last row), FAIL (+ confirmed counterexample),
    otherwise UNKNOWN."""
    assert T["u"].shape[0] == NIN and T["r"].shape[0] == 2, "expects the c=2, dims=2 table format"
    w = wave_check(T)
    if w["ok"]:
        return dict(status="PASS", wave=w, rounds_bound="3n-1")
    cex = find_counterexample(T, target) if search else None
    if cex is not None:
        return dict(status="FAIL", wave=w, counterexample=cex)
    return dict(status="UNKNOWN", wave=w)


if __name__ == "__main__":
    from . import oracle_m2d as om
    for name, T in (("K=4 oracle", om.build4()), ("K=16 oracle", om.build())):
        print(name, certify(T))
