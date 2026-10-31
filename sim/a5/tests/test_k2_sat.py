import itertools
import json
import os
import tempfile
import unittest

import numpy as np

from sim.a5 import ac, delay, k2_sat as ks, oracle_m2d as om, tasks_mul as tm
from sim.a5.tests.dpll import Dpll

SEED = 200701


def chain_instances(task, nmax, c=1):
    inst = []
    for n in range(1, nmax + 1):
        X, bits = delay.words(n, c)
        y = task(bits)
        inst += [dict(xi=X[i], tgt=y[i], mask=np.ones(n, bool)) for i in range(len(X))]
    return inst


class Checker:
    """Persistent DPLL on an encoded instance set; query(table dict) -> satisfiable with all table vars fixed?"""

    def __init__(self, cnf, tv):
        self.tv = tv
        self.s = Dpll(cnf.nv, cnf.clauses)
        assert self.s.ok
        for l in self.s.units:
            assert self.s.assign(l)
        assert self.s.propagate()
        self.base = len(self.s.trail)
        self.nv = cnf.nv

    def query(self, T):
        s = self.s
        ok = True
        for l in ks.tables_to_units(self.tv, T):
            if not s.assign(l):
                ok = False
                break
        ok = ok and s.propagate()
        if ok:  # every Tseitin/round variable must have been forced by propagation
            assert all(s.val[v] != 0 for v in range(1, self.nv + 1)), "aux variable not determined"
        s.backtrack_to(self.base)
        return ok


def random_flip(T, rng, nflips=1):
    T = {k: v.copy() for k, v in T.items()}
    for _ in range(nflips):
        kind = ["u", "r", "o"][int(rng.integers(0, 3))]
        idx = tuple(int(rng.integers(0, s)) for s in T[kind].shape)
        T[kind][idx] ^= 1
    return T


class TestCnfBuilder(unittest.TestCase):
    def test_gate_folding_and_tseitin(self):
        c = ks.Cnf()
        a, b = c.new(), c.new()
        self.assertEqual(c.and_([a, ks.TRUE]), a)
        self.assertEqual(c.and_([a, ks.FALSE]), ks.FALSE)
        self.assertEqual(c.or_([ks.FALSE, b]), b)
        self.assertEqual(c.or_([]), ks.FALSE)
        self.assertEqual(c.and_([]), ks.TRUE)
        g = c.and_([a, b])
        h = c.or_([a, b])
        for va, vb in itertools.product([0, 1], repeat=2):
            s = Dpll(c.nv, c.clauses)
            m = s.solve(assumptions=[a if va else -a, b if vb else -b])
            self.assertIsNotNone(m)
            tv_ = lambda l: (l in m) if l > 0 else (-l not in m)
            self.assertEqual(tv_(g), bool(va and vb))
            self.assertEqual(tv_(h), bool(va or vb))

    def test_empty_clause_is_unsat(self):
        c = ks.Cnf()
        c.add([ks.FALSE])
        self.assertTrue(c.unsat)
        self.assertIsNone(Dpll(c.nv, c.clauses).solve())


class TestEncoderVsInterpreter(unittest.TestCase):
    def test_exhaustive_agreement_K1_identity_task(self):
        # K=1 chain, task y = x (d=0), words n<=3: 14 table bits.  For ALL 2^14 table assignments the CNF (table vars
        # fixed, all other variables determined by propagation) is satisfiable iff ac.py reads the task correctly.
        inst = chain_instances(lambda b: delay.delay_target(b, 0), 3)
        cnf, tv, info = ks.encode(inst, 1, 4)
        self.assertEqual(sum(info["table_vars"].values()), 14)
        chk = Checker(cnf, tv)
        keys_u = sorted(tv.u)
        keys_r = sorted(tv.r)
        keys_o = sorted(tv.o)
        nbits = len(keys_u) + len(keys_r) + len(keys_o)
        by_len = {}
        for n in (1, 2, 3):
            X, bits = delay.words(n)
            by_len[n] = (X, delay.delay_target(bits, 0))
        agree = sat = 0
        for code in range(1 << nbits):
            bits_ = [(code >> i) & 1 for i in range(nbits)]
            T = dict(u=np.ones((4, 1), np.uint8), r=np.ones((1, 4, 4, 1, 1), np.uint8), o=np.zeros((4, 2), np.uint8))
            i = 0
            for k in keys_u:
                T["u"][k] = bits_[i]; i += 1
            for k in keys_r:
                T["r"][k] = bits_[i]; i += 1
            for k in keys_o:
                T["o"][k] = bits_[i]; i += 1
            ok_ac = True
            for n, (X, y) in by_len.items():
                c, _ = ac.fixed_point(T["u"], T["r"], X)
                if not np.array_equal(ac.readout(T["o"], c, X), y):
                    ok_ac = False
                    break
            ok_sat = chk.query(T)
            self.assertEqual(ok_ac, ok_sat, (code, bits_))
            agree += 1
            sat += ok_sat
        print(f"exhaustive K=1 identity task: {agree} table assignments, {sat} correct (CNF == interpreter on all)")
        self.assertGreater(sat, 0)
        self.assertLess(sat, agree)

    def test_k1_delay1_is_unsat_and_has_no_correct_table(self):
        inst = chain_instances(lambda b: delay.delay_target(b, 1), 3)
        cnf, tv, _ = ks.encode(inst, 1, 4)
        order = sorted(tv.u.values()) + sorted(tv.r.values()) + sorted(tv.o.values())
        self.assertIsNone(Dpll(cnf.nv, cnf.clauses).solve(decision_vars=order))
        # brute force over all 14-bit tables with the interpreter agrees
        keys_u, keys_r, keys_o = sorted(tv.u), sorted(tv.r), sorted(tv.o)
        nb = len(keys_u) + len(keys_r) + len(keys_o)
        by_len = {n: (delay.words(n)[0], delay.delay_target(delay.words(n)[1], 1)) for n in (1, 2, 3)}
        good = 0
        for code in range(1 << nb):
            b = [(code >> i) & 1 for i in range(nb)]
            T = dict(u=np.ones((4, 1), np.uint8), r=np.ones((1, 4, 4, 1, 1), np.uint8), o=np.zeros((4, 2), np.uint8))
            for j, k in enumerate(keys_u):
                T["u"][k] = b[j]
            for j, k in enumerate(keys_r):
                T["r"][k] = b[len(keys_u) + j]
            for j, k in enumerate(keys_o):
                T["o"][k] = b[len(keys_u) + len(keys_r) + j]
            ok = True
            for n, (X, y) in by_len.items():
                c, _ = ac.fixed_point(T["u"], T["r"], X)
                if not np.array_equal(ac.readout(T["o"], c, X), y):
                    ok = False
                    break
            good += ok
        self.assertEqual(good, 0)

    def test_planted_delay_solution_and_random_flips(self):
        rng = np.random.default_rng(SEED)
        for d, K, nmax in ((1, 3, 4), (2, 4, 4)):
            inst = chain_instances(lambda b, d=d: delay.delay_target(b, d), nmax)
            cnf, tv, _ = ks.encode(inst, K, 4, symmetry=False)
            chk = Checker(cnf, tv)
            P = delay.build(d)
            self.assertEqual(ks.verify(P, inst), 0)
            self.assertTrue(chk.query(P))
            nsat = nunsat = 0
            for _ in range(40):
                T = random_flip(P, rng, 1 + int(rng.integers(0, 3)))
                want = ks.verify(T, inst) == 0
                self.assertEqual(chk.query(T), want)
                nsat += want
                nunsat += not want
            print(f"delay d={d} K={K}: planted + 40 random flips -> {nsat} still correct, {nunsat} broken; CNF agrees on all")
            self.assertGreater(nunsat, 0)

    def test_m2d_encoding_vs_interpreter_with_k4_oracle(self):
        rng = np.random.default_rng(SEED + 1)
        for n, nq in ((1, 25), (2, 4)):
            inst = ks.m2d_instances(n, "all")
            cnf, tv, info = ks.encode(inst, 4, 8, symmetry=False)
            chk = Checker(cnf, tv)
            P = om.build4()
            self.assertEqual(ks.verify(P, inst), 0)
            self.assertTrue(chk.query(P))
            res = []
            for _ in range(nq):
                T = random_flip(P, rng, 1)
                # make flips land on entries the instances read at least half of the time
                want = ks.verify(T, inst) == 0
                self.assertEqual(chk.query(T), want)
                res.append(want)
            print(f"M2D n={n} K=4 'all' targets: vars={info['vars']} clauses={info['clauses']}; flips correct={sum(res)}/{len(res)}")

    def test_fixed_point_constraint_when_rounds_truncated(self):
        # rounds=1 on a chain: tables that need more rounds to converge are rejected (d[2] != d[1]) even though ac would be fine later
        inst = chain_instances(lambda b: delay.delay_target(b, 1), 3)
        cnf, tv, _ = ks.encode(inst, 3, 4, rounds=1, symmetry=False)
        chk = Checker(cnf, tv)
        P = delay.build(1)
        self.assertEqual(ks.verify(P, inst), 0)
        self.assertFalse(chk.query(P))  # needs >1 round to converge on length-3 words


def canonicalize(T, x0):
    """Relabel latent values so that u[x0] is non-increasing in h (the symmetry-breaking normal form)."""
    perm_src = sorted(range(T["u"].shape[1]), key=lambda h: -int(T["u"][x0, h]))  # new position i holds old value perm_src[i]
    K = len(perm_src)
    new_of_old = {old: new for new, old in enumerate(perm_src)}
    u = T["u"][:, perm_src]
    r = T["r"][:, :, :, perm_src][:, :, :, :, perm_src]
    o = np.zeros_like(T["o"])
    for S in range(1 << K):
        S2 = sum(1 << new_of_old[h] for h in range(K) if (S >> h) & 1)
        o[:, S2] = T["o"][:, S]
    return dict(u=u, r=r, o=o)


class TestSymmetryBreaking(unittest.TestCase):
    def test_canonical_planted_solutions_satisfy_symmetry_clauses(self):
        for d, K, nmax in ((1, 3, 4), (2, 4, 4)):
            inst = chain_instances(lambda b, d=d: delay.delay_target(b, d), nmax)
            cnf, tv, _ = ks.encode(inst, K, 4, symmetry=True)
            P = delay.build(d)
            self.assertFalse(Checker(cnf, tv).query(P))     # raw construction violates the normal form ...
            C = canonicalize(P, 2)
            self.assertEqual(ks.verify(C, inst), 0)          # ... relabelling keeps it correct ...
            self.assertTrue(Checker(cnf, tv).query(C))       # ... and the relabelled table satisfies the full CNF


class TestSolveToy(unittest.TestCase):
    def test_k2_delay1_sat_decode_verify(self):
        inst = chain_instances(lambda b: delay.delay_target(b, 1), 3)
        cnf, tv, info = ks.encode(inst, 2, 4)
        order = sorted(tv.u.values()) + sorted(tv.r.values()) + sorted(tv.o.values())
        model = Dpll(cnf.nv, cnf.clauses).solve(decision_vars=order)
        self.assertIsNotNone(model, "K=2 can represent d=1 (hand construction exists), so the CNF must be SAT")
        mapping = json.loads(json.dumps(tv.mapping(4, 1)))  # round trip through JSON like the CLI
        T = ks.decode(mapping, set(model))
        self.assertEqual(ks.verify(T, inst), 0)
        # does the solver's solution generalise to longer words? report only (not guaranteed by the training set)
        rng = np.random.default_rng(SEED + 2)
        wrong = 0
        for n in range(4, 9):
            X, y = delay.gen(rng, 100, n, 1)
            ii = [dict(xi=X[i], tgt=y[i], mask=np.ones(n, bool)) for i in range(100)]
            wrong += ks.verify(T, ii)
        print("K=2 d=1 toy solution (DPLL) wrong on", wrong, "of 500 longer words (n=4..8)")

    def test_unsat_toy_pigeonhole_consistent(self):
        # d=3 with K=1 violates K >= d-1 = 2: must be UNSAT for words up to length 6 (via k2_sat)
        inst = chain_instances(lambda b: delay.delay_target(b, 3), 6)
        cnf, tv, _ = ks.encode(inst, 1, 4)
        order = sorted(tv.u.values()) + sorted(tv.r.values()) + sorted(tv.o.values())
        self.assertIsNone(Dpll(cnf.nv, cnf.clauses).solve(decision_vars=order))


class TestKMinD2(unittest.TestCase):
    def test_d2_k1_unsat_so_kmin_ge_2(self):
        # words up to length 5 already force UNSAT for K=1, hence K_min(d=2) >= 2 (a subset of inputs is UNSAT => all is)
        inst = chain_instances(lambda b: delay.delay_target(b, 2), 5)
        cnf, tv, _ = ks.encode(inst, 1, 4)
        order = sorted(tv.u.values()) + sorted(tv.r.values()) + sorted(tv.o.values())
        self.assertIsNone(Dpll(cnf.nv, cnf.clauses).solve(decision_vars=order))


class TestDimacsAndCli(unittest.TestCase):
    def test_dimacs_roundtrip_and_cli(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.cnf")
            ks.cli(["--task", "m2d", "--n", "1", "--K", "2", "--out", path, "--no-solve"])
            with open(path) as f:
                lines = f.read().splitlines()
            head = [l for l in lines if l.startswith("p cnf")][0].split()
            nv, nc = int(head[2]), int(head[3])
            body = [l for l in lines if not l.startswith(("c", "p"))]
            self.assertEqual(len(body), nc)
            self.assertTrue(all(abs(int(t)) <= nv for l in body for t in l.split()))
            with open(path + ".map.json") as f:
                mapping = json.load(f)
            self.assertEqual(mapping["K"], 2)
            # decode path from a solver-style output file with an arbitrary model (all vars false)
            mf = os.path.join(td, "model.txt")
            with open(mf, "w") as f:
                f.write("s SATISFIABLE\nv -1 -2 -3 0\n")
            res = ks.cli(["--task", "m2d", "--n", "1", "--out", path, "--no-solve", "--decode", mf])
            self.assertEqual(res, "SAT")  # decode reports (the decoded all-false tables are of course wrong; only plumbing is tested)

    def test_cli_writes_result_json(self):
        import sys
        import types
        for status in ("UNSAT", "TIMEOUT", "SAT"):
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "k.cnf")
                old_mod, old_solve = sys.modules.get("pysat"), ks.solve_pysat
                sys.modules["pysat"] = types.ModuleType("pysat")
                ks.solve_pysat = lambda p, solver, tl, st=status: (st, [] if st == "SAT" else None)
                try:
                    ks.cli(["--task", "delay", "--d", "1", "--n", "2", "--K", "2", "--time-limit", "7", "--out", path])
                finally:
                    ks.solve_pysat = old_solve
                    if old_mod is None:
                        del sys.modules["pysat"]
                    else:
                        sys.modules["pysat"] = old_mod
                with open(path + ".result.json") as f:
                    r = json.load(f)
                self.assertEqual(r["status"], status)
                self.assertEqual((r["task"], r["n"], r["K"], r["d"]), ("delay", 2, 2, 1))
                for k in ("targets", "vars", "clauses", "seconds"):
                    self.assertIn(k, r)
                self.assertGreater(r["vars"], 0)

    def test_m2d_k2_n1_sizes(self):
        inst = ks.m2d_instances(1)
        cnf, tv, info = ks.encode(inst, 2, 8)
        self.assertEqual(info["instances"], 4)
        self.assertEqual(info["table_vars"]["o"], 16)


if __name__ == "__main__":
    unittest.main()
