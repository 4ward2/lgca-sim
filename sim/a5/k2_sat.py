"""SAT encoding of  "exists a K-latent-value table set (u, r, o) that is correct on every given instance"
(sim/a5/ac.py semantics = b_arm.TernaryNet).  Default use: M2D, K = 2, n <= 3 (Fable review section 2.1).  Generalised to
any K and to chains (delay line, see delay.py).

Encoding (everything is exact, no relaxation):
  table variables   u[x][h], r[axis][xa][xb][ha][hb]  (xa/xb the codes of the first/second cell along the axis, code 0 =
                    virtual cell), o[x][S] for every subset S of the K latent values.  Only entries the instances read
                    get a variable; all other entries are irrelevant to the instances.
  per instance      d[t][cell][h] = "value h still possible after t synchronous AC rounds", d[0] = true, d[t+1] = F(d[t]) with
                      F(d)[v][h] = u[x_v][h]  AND  AND_{ports w}  OR_g ( r_port(g, h)  AND  d[w][g] )
                    (virtual neighbour: d = true).  Both directions of "=" are encoded (Tseitin), the final set must
                    be the exact fixed point.  d[t+1] => d[t] is added as a redundant monotonicity clause.
  rounds T          chains: T = N (proof: F <= F_left, F_right, a one-pass left/right scan is exact after N rounds, so
                    d[N] = gfp).  2-D: T = N*K (each changing round removes >= 1 of the N*K values, so d[N*K] = gfp).  A smaller
                    --rounds T adds the constraint d[T+1] = d[T] (a converged-within-T-rounds restriction: SAT is still
                    a genuine solution, but UNSAT is then only "no solution converging within T rounds").
  K = 2 (Lemma 2): AC is implication-graph reachability; the round bound N*K = 2N is exactly the longest possible
                    implication path, which is why T = 2N is the sound compact horizon in that case.
  readout           for every scored cell with final bits f and target t:  for each S in 2^K :
                    (f == S)  =>  (o[x][S] == t)   (one clause per S).
  symmetry          latent relabelling is a symmetry of all tables; we sort u[x0] (x0 = smallest real code) non-increasingly
                    in h (sound: any solution can be relabelled into this form).  --no-symmetry disables it.
Solving: DIMACS is written (`--out`), and if `pysat` is importable it is solved in a worker process with a wall-clock
limit (`--time-limit`).  Any DIMACS solver (kissat, cadical) works; `--decode model.txt` turns its output into tables.
Outcome meaning (M2D, K=2): UNSAT => no K=2 table set is correct on all n x 2n inputs (theorem for that n and target);
SAT => the decoded tables are verified with ac.py on the instances and tested on larger n (conjecture refuted if they hold).
"""
import argparse
import itertools
import json
import os
import sys
import tempfile
import time

import numpy as np

from . import ac

TRUE, FALSE = "T", "F"


def neg(l):
    return FALSE if l == TRUE else TRUE if l == FALSE else -l


class Cnf:
    def __init__(self):
        self.nv = 0
        self.clauses = []
        self.unsat = False

    def new(self):
        self.nv += 1
        return self.nv

    def add(self, lits):
        out = []
        for l in lits:
            if l == TRUE:
                return
            if l != FALSE:
                out.append(l)
        if not out:
            self.unsat = True
        self.clauses.append(out)

    def and_(self, lits):
        out = []
        for l in lits:
            if l == FALSE:
                return FALSE
            if l != TRUE:
                out.append(l)
        if not out:
            return TRUE
        if len(out) == 1:
            return out[0]
        t = self.new()
        for l in out:
            self.clauses.append([-t, l])
        self.clauses.append([t] + [-l for l in out])
        return t

    def or_(self, lits):
        return neg(self.and_([neg(l) for l in lits]))

    def write(self, path, comments=()):
        with open(path, "w") as f:
            for c in comments:
                f.write("c " + c + "\n")
            f.write(f"p cnf {self.nv} {len(self.clauses)}\n")
            for cl in self.clauses:
                f.write(" ".join(map(str, cl)) + " 0\n")


class TableVars:
    """Lazily allocated table variables."""

    def __init__(self, cnf, K):
        self.cnf, self.K = cnf, K
        self.u, self.r, self.o = {}, {}, {}

    def _get(self, d, key):
        if key not in d:
            d[key] = self.cnf.new()
        return d[key]

    def U(self, x, h):
        return self._get(self.u, (x, h))

    def R(self, ax, xa, xb, ha, hb):
        return self._get(self.r, (ax, xa, xb, ha, hb))

    def O(self, x, S):
        return self._get(self.o, (x, S))

    def mapping(self, nin, dims):
        return dict(K=self.K, nin=nin, dims=dims,
                    u=[list(k) + [v] for k, v in self.u.items()], r=[list(k) + [v] for k, v in self.r.items()],
                    o=[list(k) + [v] for k, v in self.o.items()])


def _ports(dims):
    """(offset tuple, axis, neighbour_is_first)."""
    if dims == 1:
        return [((-1,), 0, True), ((1,), 0, False)]
    return [((-1, 0), 0, True), ((1, 0), 0, False), ((0, -1), 1, True), ((0, 1), 1, False)]


def encode(instances, K, nin, rounds=None, symmetry=True):
    """instances: list of dict(xi=int array (N,) or (H, W), tgt=same shape, mask=bool same shape).
    Returns (cnf, tabvars, info)."""
    dims = instances[0]["xi"].ndim
    cnf = Cnf()
    tv = TableVars(cnf, K)
    ports = _ports(dims)
    full_nbr_rounds = False
    for inst in instances:
        xi, tgt, mask = inst["xi"], inst["tgt"], inst["mask"]
        cells = list(np.ndindex(*xi.shape))
        N = len(cells)
        T = rounds or (N if dims == 1 else N * K)
        need_fix = T < (N if dims == 1 else N * K)
        # neighbour table: per cell list of (nbr cell or None, axis, nbr_is_first)
        nbrs = {}
        for c in cells:
            lst = []
            for off, ax, first in ports:
                nb = tuple(a + b for a, b in zip(c, off))
                ok = all(0 <= nb[i] < xi.shape[i] for i in range(dims))
                lst.append((nb if ok else None, ax, first))
            nbrs[c] = lst
        d = {c: [TRUE] * K for c in cells}

        def step(dprev):
            dn = {}
            for c in cells:
                x = int(xi[c])
                new = []
                for h in range(K):
                    terms = [tv.U(x, h)]
                    for nb, ax, first in nbrs[c]:
                        xn = int(xi[nb]) if nb is not None else 0
                        dom = dprev[nb] if nb is not None else [TRUE] * K
                        sup = []
                        for g in range(K):
                            Rl = tv.R(ax, xn, x, g, h) if first else tv.R(ax, x, xn, h, g)
                            sup.append(cnf.and_([Rl, dom[g]]))
                        terms.append(cnf.or_(sup))
                    new.append(cnf.and_(terms))
                dn[c] = new
            return dn

        for t in range(T):
            dn = step(d)
            for c in cells:
                for h in range(K):
                    cnf.add([neg(dn[c][h]), d[c][h]])  # monotone (redundant, helps propagation)
            d = dn
        if need_fix:
            dn = step(d)
            for c in cells:
                for h in range(K):
                    a, b = d[c][h], dn[c][h]
                    cnf.add([neg(a), b])
                    cnf.add([a, neg(b)])
        for c in cells:
            if not mask[c]:
                continue
            x, t = int(xi[c]), int(tgt[c])
            f = d[c]
            for S in range(1 << K):
                ol = tv.O(x, S)
                cl = [(neg(f[h]) if (S >> h) & 1 else f[h]) for h in range(K)]
                cl.append(ol if t else neg(ol))
                cnf.add(cl)
    if symmetry:
        real = sorted({int(v) for inst in instances for v in inst["xi"].reshape(-1)})
        x0 = real[0]
        for h in range(K - 1):
            cnf.add([neg(tv.U(x0, h + 1)), tv.U(x0, h)])
    info = dict(instances=len(instances), K=K, nin=nin, dims=dims, vars=cnf.nv, clauses=len(cnf.clauses),
                table_vars=dict(u=len(tv.u), r=len(tv.r), o=len(tv.o)), symmetry=symmetry, rounds=rounds or "N (chain) / N*K (2-D)")
    return cnf, tv, info


# ---------------------------------------------------------------------------------------------- decode / verify
def tables_to_units(tv, T):
    """Literals fixing every allocated table variable to the value in the table dict T (for tests)."""
    lits = []
    for (x, h), v in tv.u.items():
        lits.append(v if T["u"][x, h] else -v)
    for (ax, xa, xb, ha, hb), v in tv.r.items():
        lits.append(v if T["r"][ax, xa, xb, ha, hb] else -v)
    for (x, S), v in tv.o.items():
        lits.append(v if T["o"][x, S] else -v)
    return lits


def decode(mapping, true_vars):
    """mapping from TableVars.mapping (or JSON), true_vars: set of positive var ids assigned true.
    Unrequested entries: u = 1, r = 1, o = 0."""
    K, nin, dims = mapping["K"], mapping["nin"], mapping["dims"]
    u = np.ones((nin, K), np.uint8)
    r = np.ones((dims, nin, nin, K, K), np.uint8)
    o = np.zeros((nin, 1 << K), np.uint8)
    for x, h, v in mapping["u"]:
        u[x, h] = v in true_vars
    for ax, xa, xb, ha, hb, v in mapping["r"]:
        r[ax, xa, xb, ha, hb] = v in true_vars
    for x, S, v in mapping["o"]:
        o[x, S] = v in true_vars
    return dict(u=u, r=r, o=o)


def verify(T, instances):
    """Number of instances (of the given list) on which ac.py's fixed-point readout is wrong on a scored cell."""
    bad = 0
    for inst in instances:
        xi = inst["xi"][None]
        c, _ = ac.fixed_point(T["u"], T["r"], xi)
        y = ac.readout(T["o"], c, xi)[0]
        bad += int((y[inst["mask"]] != inst["tgt"][inst["mask"]]).any())
    return bad


# ---------------------------------------------------------------------------------------------- instances
def m2d_instances(n, targets="last"):
    from . import tasks_mul as tm
    d = tm.generate_all("M2D", n)
    xi = ac.codes_from_x(d["x"])
    y = d["y"].astype(np.int64)
    mask = d["mask"] if targets == "last" else d["mask_all"]
    return [dict(xi=xi[i], tgt=y[i], mask=mask[i]) for i in range(len(xi))]


def delay_instances(d, nmax):
    from . import delay
    return delay.sat_instances(d, nmax)


# ---------------------------------------------------------------------------------------------- solving
def _worker(path, solver, q):
    from pysat.formula import CNF
    from pysat.solvers import Solver
    cnf = CNF(from_file=path)
    with Solver(name=solver, bootstrap_with=cnf.clauses) as s:
        if s.solve():
            q.put(("SAT", [l for l in s.get_model() if l > 0]))
        else:
            q.put(("UNSAT", None))


def solve_pysat(path, solver="cadical153", time_limit=0):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(path, solver, q))
    p.start()
    try:
        res = q.get(timeout=time_limit or None)
    except Exception:
        res = ("TIMEOUT", None)
    p.terminate()
    p.join()
    return res


def parse_model_file(path):
    """kissat / cadical / minisat style: lines 's SATISFIABLE', 'v 1 -2 ... 0'  (or plain minisat 'SAT\\n1 -2 ...')."""
    pos = set()
    with open(path) as f:
        for line in f:
            if line.startswith("v ") or (line[:1] in "-0123456789"):
                for tok in line.replace("v ", "", 1).split():
                    t = int(tok)
                    if t > 0:
                        pos.add(t)
    return pos


def build_instances(a):
    if a.task == "m2d":
        return m2d_instances(a.n, a.targets), 8
    return delay_instances(a.d, a.n), 4


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=["m2d", "delay"], default="m2d")
    ap.add_argument("--n", type=int, required=True, help="m2d: grid size n x 2n;  delay: max word length")
    ap.add_argument("--d", type=int, default=1, help="delay d")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--targets", choices=["last", "all"], default="last")
    ap.add_argument("--rounds", type=int, default=0, help="0 = sound default (N for chains, N*K for 2-D)")
    ap.add_argument("--out", default=None, help="DIMACS path (a .map.json sidecar is written next to it)")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds; 0 = unlimited")
    ap.add_argument("--solver", default="cadical153")
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--no-solve", action="store_true")
    ap.add_argument("--decode", default=None, help="solver output file; requires --out (for the .map.json)")
    a = ap.parse_args(argv)
    inst, nin = build_instances(a)
    t0 = time.time()
    cnf, tv, info = encode(inst, a.K, nin, rounds=a.rounds or None, symmetry=not a.no_symmetry)
    info.update(task=a.task, n=a.n, d=a.d if a.task == "delay" else None, targets=a.targets,
                encode_seconds=round(time.time() - t0, 1))
    print(json.dumps(info))
    mapping = tv.mapping(nin, inst[0]["xi"].ndim)
    path = a.out
    if path is None and not a.no_solve:
        path = os.path.join(tempfile.mkdtemp(), "k2.cnf")
    if path:
        cnf.write(path, comments=[json.dumps(info)])
        with open(path + ".map.json", "w") as f:
            json.dump(mapping, f)
        print("wrote", path)
    if a.decode:
        sol = decode(mapping, parse_model_file(a.decode))
        return report_solution(sol, inst, a)
    if a.no_solve:
        return None
    try:
        import pysat  # noqa: F401
    except ImportError:
        print("pysat not importable: not solving (run kissat/cadical on the DIMACS, then --decode)")
        return None
    status, model = solve_pysat(path, a.solver, a.time_limit)
    print("RESULT", status)
    if status == "SAT":
        return report_solution(decode(mapping, set(model)), inst, a)
    return status


def report_solution(sol, inst, a):
    bad = verify(sol, inst)
    print(f"decoded tables: wrong on {bad} of {len(inst)} training instances (must be 0)")
    out = {k: v.tolist() for k, v in sol.items()}
    outpath = (a.out or "k2") + ".sol.json"
    with open(outpath, "w") as f:
        json.dump(out, f)
    print("tables written to", outpath)
    if a.task == "m2d":
        for n in range(a.n + 1, 9):
            ii = m2d_instances(n, a.targets) if n <= 5 else _random_m2d(n, a.targets)
            print(f"  generalisation n={n}: wrong on {verify(sol, ii)} of {len(ii)}")
    else:
        from . import delay
        rng = np.random.default_rng(200701)
        for n in range(a.n + 1, a.n + 6):
            X, y = delay.gen(rng, 300, n, a.d)
            ii = [dict(xi=X[i], tgt=y[i], mask=np.ones(n, bool)) for i in range(len(X))]
            print(f"  generalisation n={n}: wrong on {verify(sol, ii)} of {len(ii)}")
    return "SAT"


def _random_m2d(n, targets):
    from . import tasks_mul as tm
    rng = np.random.default_rng(200702 + n)
    d = tm.generate("M2D", rng, 300, n)
    xi = ac.codes_from_x(d["x"])
    mask = d["mask"] if targets == "last" else d["mask_all"]
    return [dict(xi=xi[i], tgt=d["y"][i].astype(np.int64), mask=mask[i]) for i in range(len(xi))]


if __name__ == "__main__":
    cli()
