"""All-length certificate for the delay-line task y_v = x_{v-d} on a chain (ported from Fable's prototype
reviews/a5/fable_scratch2/delay_cert_proto.py; same argument as sim/cert/chain.py, bimachine pair check).

On a chain the AC fixed point is S_v = L_v & R_v (L: left-to-right support scan, R: right-to-left).  The left automaton has
states (x_v, L_v, window of the last d bits incl. x_v, expected output y_v = x_{v-d}); the right automaton has states
(x_v, R_v).  Every (left, right) pair with the same letter is realised by a word  prefix + suffix[1:]  (any prefix reaching the
left state, any suffix reaching the right state), and the readout there is o[x][L & R].  Hence

    PASS    <=> for every reachable pair with equal letter  o[x][L & R] == y   (theorem for ALL chain lengths),
    FAIL     : some pair has o[x][L & R] != y; the shortest such word is returned and CONFIRMED by the ac.py interpreter
               (readout differs from the target) and cross-checked with the frozen TernaryNet (simulator_agrees);
    UNKNOWN  : the search found a violating pair but the interpreter did not confirm it (should not happen; reported
               rather than hidden), or the automaton exceeded max_states.

Tables: sim/a5/ac.py format with c = 1 (nin = 4, real codes 2 + bit, virtual code 0), any K.
"""
from collections import deque

import numpy as np

from . import chain_utils as cu, delay


def certify(T, d, c=1, max_states=2_000_000):
    u, r = T["u"], T["r"]
    K = u.shape[1]
    full = (1 << K) - 1
    um = cu.u_masks(u)
    rm = cu.row_masks(r[0])
    rmT = cu.row_masks(np.swapaxes(r[0], -1, -2))  # rmT[xa, xb, g] = mask over h with r[xa, xb, h, g]

    def supp(rmx, xa, xb, S):
        out = 0
        for g in range(K):
            if (S >> g) & 1:
                out |= int(rmx[xa, xb, g])
        return out

    letters = [delay.code_of(b, c) for b in (0, 1)]
    wmask = (1 << d) - 1
    left, q = {}, deque()
    for x in letters:
        st = (x, int(um[x]) & supp(rm, 0, x, full), x & 1 & wmask, 0)  # y_0 = 0 (v < d)
        if st not in left:
            left[st] = [x]
            q.append(st)
    while q:
        st = q.popleft()
        x, L, w, _ = st
        for z in letters:
            y = (w >> (d - 1)) & 1
            ns = (z, int(um[z]) & supp(rm, x, z, L), ((w << 1) | (z & 1)) & wmask, y)
            if ns not in left:
                if len(left) > max_states:
                    return dict(status="UNKNOWN", reason="left automaton too large", left_states=len(left))
                left[ns] = left[st] + [z]
                q.append(ns)
    right, q = {}, deque()
    for x in letters:
        st = (x, int(um[x]) & supp(rmT, x, 0, full))
        if st not in right:
            right[st] = [x]
            q.append(st)
    while q:
        st = q.popleft()
        x, R = st
        for z in letters:
            ns = (z, int(um[z]) & supp(rmT, z, x, R))
            if ns not in right:
                if len(right) > max_states:
                    return dict(status="UNKNOWN", reason="right automaton too large", right_states=len(right))
                right[ns] = [z] + right[st]
                q.append(ns)
    by_letter = {}
    for (z, R), suf in right.items():
        by_letter.setdefault(z, []).append((R, suf))
    best = None
    for (x, L, w, y), pre in left.items():
        for R, suf in by_letter.get(x, []):
            if int(T["o"][x, L & R]) != y:
                word = pre + suf[1:]
                if best is None or len(word) < len(best):
                    best = word
    out = dict(left_states=len(left), right_states=len(right))
    if best is None:
        out["status"] = "PASS"
        return out
    X = np.array(best, np.int64)[None]
    pred = cu.predict(T, X)[0]
    want = delay.delay_target(X & 1, d)[0]
    confirmed = bool((pred != want).any())
    out.update(status="FAIL" if confirmed else "UNKNOWN", counterexample=best, pred=pred.tolist(), want=want.tolist(),
               interpreter_confirms=confirmed,
               simulator_agrees=bool(np.array_equal(cu.ternary_predict(T, X[0]), pred)))
    return out


if __name__ == "__main__":
    for dd in (1, 2, 3, 4):
        print("build", dd, certify(delay.build(dd), dd)["status"])
