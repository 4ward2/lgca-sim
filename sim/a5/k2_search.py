"""Is there ANY K=2 table set (u, r, o) that is correct on every M2D input of an n x 2n grid?

Same semantics as sim/b_arm.TernaryNet (interpreter: sim/a5/ac.py).  Correct = the readout equals the product bit at
every cell of the last row (the task output), for all 4^n inputs, after the synchronous-AC fixed point.

Reductions (all sound):
  * only table entries that the n x 2n grid can ever read are variables (u of the real codes; r[axis][xa][xb] for the
    code pairs that occur as neighbours, including virtual code 0 at the border);
  * o is not searched: given (u, r) a correct o exists iff no (code, final-set) key is demanded to be both 0 and 1
    (conflict test), so o is eliminated;
  * global latent relabelling h0<->h1 maps solutions to solutions (all tables shared): require u <= swap(u) lexicographically;
  * branch and bound: unassigned bits are set to 0 (lower-bound sets) and to 1 (upper-bound sets); AC is monotone in
    the tables, so S_lo <= S_final <= S_hi.  Where S_lo == S_hi at two scored cells with the same code but different
    targets the partial assignment is dead.
Modes:
  size    print the search-space size
  dfs     branch and bound (--time-limit seconds; --shard i/m splits the work by the first --split-bits decisions)
  anneal  random-restart local search minimising the number of cell instances that no o can classify (diagnostic only;
          it can find a solution but cannot prove absence)
"""
import argparse
import itertools
import json
import random
import time

import numpy as np

from . import ac, tasks_mul as tm

K = 2
NIN = 8


def problem(n):
    d = tm.generate_all("M2D", n)
    xi = ac.codes_from_x(d["x"])
    tgt = d["y"][:, -1, :].astype(np.int64)  # (B, 2n) last row = task output
    H, W = xi.shape[1:]
    xp = np.pad(xi, [(0, 0), (1, 1), (1, 1)])
    pairs = set()
    for j in range(H + 1):  # vertical neighbours (padded rows j, j+1), real columns
        for k in range(W):
            pairs.update(zip(xp[:, j, k + 1].tolist(), xp[:, j + 1, k + 1].tolist()))
    pairs = {(0,) + p for p in pairs}
    h = set()
    for j in range(H):
        for k in range(W + 1):
            h.update(zip(xp[:, j + 1, k].tolist(), xp[:, j + 1, k + 1].tolist()))
    pairs |= {(1,) + p for p in h}
    pairs = sorted(p for p in pairs if (p[1] >> 2) or (p[2] >> 2))  # at least one real cell
    pairs.sort(key=lambda p: (p[1] != 0 and p[2] != 0, p))  # border relations first
    real = sorted(set(xi.reshape(-1).tolist()))
    vars_ = [("u", (x, h_)) for x in real for h_ in range(K)]
    for ax, xa, xb in pairs:
        for ha in range(K):
            for hb in range(K):
                vars_.append(("r", (ax, xa, xb, ha, hb)))
    return dict(n=n, xi=xi, tgt=tgt, real=real, pairs=pairs, vars=vars_)


def space_size(P):
    nu = len(P["real"]) * K
    nr = K * K * len(P["pairs"])
    full = NIN * K + 2 * NIN * NIN * K * K + NIN * (1 << K)
    return dict(n=P["n"], inputs=int(P["xi"].shape[0]), real_codes=len(P["real"]), pairs_read=len(P["pairs"]),
                u_bits=nu, r_bits=nr, log2_space_searched=nu + nr, log2_space_before_symmetry_o_elim_unread=full)


def make_tables(val):
    """val: dict var -> 0/1.  Returns [(u, r) with unassigned=0, (u, r) with unassigned=1]."""
    out = []
    for fill in (0, 1):
        u = np.full((NIN, K), fill, np.uint8)
        r = np.full((2, NIN, NIN, K, K), fill, np.uint8)
        for (kind, idx), b in val.items():
            if kind == "u":
                u[idx] = b
            else:
                r[idx] = b
        out.append((u, r))
    return out


def final_sets(P, u, r):
    c, _ = ac.fixed_point(u, r, P["xi"])
    return ac.subset_index(c)[:, -1, :]  # (B, 2n) last row


def conflict(P, lo_sets, hi_sets):
    """True if two last-row cells with equal code and determined equal set demand different bits."""
    det = lo_sets == hi_sets
    codes = P["xi"][:, -1, :]
    seen = {}
    for b, k in zip(*np.nonzero(det)):
        key = (int(codes[b, k]), int(lo_sets[b, k]))
        t = int(P["tgt"][b, k])
        if seen.setdefault(key, t) != t:
            return True
    return False


def misclassified(P, sets):
    """Number of last-row cell instances misclassified by the best o, for COMPLETE tables (0 = correct model)."""
    codes = P["xi"][:, -1, :]
    cnt = {}
    for b in range(sets.shape[0]):
        for k in range(sets.shape[1]):
            c = cnt.setdefault((int(codes[b, k]), int(sets[b, k])), [0, 0])
            c[int(P["tgt"][b, k])] += 1
    return sum(min(c) for c in cnt.values())


def sym_ok(P, val):
    """u <= swap(u) lexicographically, checked once u is fully assigned (relabelling h0<->h1)."""
    real = P["real"]
    if not all((("u", (x, h)) in val) for x in real for h in range(K)):
        return True
    a = [val[("u", (x, h))] for x in real for h in range(K)]
    b = [val[("u", (x, 1 - h))] for x in real for h in range(K)]
    return a <= b


def alive(P, val):
    if not sym_ok(P, val):
        return False
    (ulo, rlo), (uhi, rhi) = make_tables(val)
    return not conflict(P, final_sets(P, ulo, rlo), final_sets(P, uhi, rhi))


def dfs(P, time_limit, shard=(0, 1), split_bits=12, log=print):
    vars_ = P["vars"]
    t0 = time.time()
    stats = dict(nodes=0, pruned=0, max_depth=0, solutions=0, timeout=False, prefixes_assigned=0, prefixes_done=0)
    sols = []
    val = {}

    def rec(i):
        if time.time() - t0 > time_limit:
            stats["timeout"] = True
            return
        stats["nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], i)
        if i == len(vars_):
            stats["solutions"] += 1
            sols.append({f"{k}{idx}": b for (k, idx), b in val.items()})
            return
        v = vars_[i]
        for b in (0, 1):
            val[v] = b
            if alive(P, val):
                rec(i + 1)
            else:
                stats["pruned"] += 1
            del val[v]
            if stats["timeout"]:
                return

    split_bits = min(split_bits, len(vars_))
    for pi, bits_ in enumerate(itertools.product((0, 1), repeat=split_bits)):
        if pi % shard[1] != shard[0]:
            continue
        stats["prefixes_assigned"] += 1
        val.clear()
        val.update(zip(vars_[:split_bits], bits_))
        if alive(P, val):
            rec(split_bits)
            if stats["timeout"]:
                break
        stats["prefixes_done"] += 1
        if stats["prefixes_done"] % 16 == 0:
            log(f"[{time.time() - t0:.0f}s] prefixes done {stats['prefixes_done']} nodes {stats['nodes']}")
    stats["elapsed"] = round(time.time() - t0, 1)
    stats["exhausted"] = not stats["timeout"]
    return stats, sols


def anneal(P, time_limit, seed, log=print):
    rng = random.Random(seed)
    vars_ = P["vars"]
    t0 = time.time()
    best = None
    restarts = 0

    def score(val):
        (u, r), _ = make_tables(val)
        return misclassified(P, final_sets(P, u, r))

    while time.time() - t0 < time_limit and (best is None or best[0] > 0):
        restarts += 1
        val = {v: rng.randint(0, 1) for v in vars_}
        cur, T = score(val), 2.0
        for _ in range(400):
            if time.time() - t0 > time_limit or cur == 0:
                break
            v = rng.choice(vars_)
            val[v] ^= 1
            s = score(val)
            if s <= cur or rng.random() < np.exp((cur - s) / T):
                cur = s
            else:
                val[v] ^= 1
            T = max(0.2, T * 0.99)
        if best is None or cur < best[0]:
            best = (cur, dict(val))
            log(f"[{time.time() - t0:.0f}s] restart {restarts}: best misclassified {cur} of {int(P['tgt'].size)}")
    return dict(best_misclassified=best[0], total_instances=int(P["tgt"].size), restarts=restarts,
                elapsed=round(time.time() - t0, 1)), best[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["size", "dfs", "anneal"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds")
    ap.add_argument("--shard", default="0/1", help="i/m")
    ap.add_argument("--split-bits", type=int, default=12)
    ap.add_argument("--seed", type=int, default=200101)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    P = problem(a.n)
    size = space_size(P)
    print("space:", json.dumps(size))
    res = dict(space=size)
    if a.mode == "dfs":
        i, m = (int(t) for t in a.shard.split("/"))
        stats, sols = dfs(P, a.time_limit, (i, m), a.split_bits)
        print("dfs:", json.dumps(stats))
        res.update(dfs=stats, solutions=sols[:20])
    elif a.mode == "anneal":
        stats, _ = anneal(P, a.time_limit, a.seed)
        print("anneal:", json.dumps(stats))
        res.update(anneal=stats)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
    return res


if __name__ == "__main__":
    main()
