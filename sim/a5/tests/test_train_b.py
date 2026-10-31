import json
import os
import tempfile
import unittest

import numpy as np

from sim.a5 import m1d_cert, oracle_m2d as om, train_b as tb


class TestRunAc(unittest.TestCase):
    def test_m2d_oracle_is_exact_and_halts(self):
        tabs = om.build()
        for task in ("m2d_dense", "m2d_last"):
            for n in (3, 5, 8):
                d = tb.fixed_set(task, n, 60, 300555)
                s = tb.summarize(tb.run_ac(tabs, d))
                self.assertEqual(s["exact_fp"], 1.0, (task, n, s))
                self.assertEqual(s["exact_halt"], 1.0, (task, n, s))
                self.assertEqual(s["fp_conv_rate"], 1.0)

    def test_broken_table_fails(self):
        tabs = {k: v.copy() for k, v in om.build().items()}
        tabs["o"][:] = 0
        d = tb.fixed_set("m2d_dense", 4, 40, 300556)
        self.assertLess(tb.summarize(tb.run_ac(tabs, d))["exact_halt"], 0.5)

    def test_smoke_train_writes_outputs(self):
        with tempfile.TemporaryDirectory() as td_:
            tb.main(["--task", "delay3", "--K", "5", "--seed", "300901", "--steps", "4", "--eval_every", "2",
                     "--n_dev", "20", "--n_test", "20", "--out", td_])
            for f in ("config.json", "final.json", "test.json", "tables.json", "log.json"):
                self.assertTrue(os.path.exists(os.path.join(td_, f)), f)
            with open(os.path.join(td_, "test.json"), encoding="utf-8") as f:
                t = json.load(f)
            self.assertEqual(sorted(t["per_size"]), ["128", "32", "64"])


class TestHaltingSemantics(unittest.TestCase):
    def test_simulation_capped_at_tmax_no_extension(self):
        tabs = om.build4()
        d = tb.fixed_set("m2d_dense", 5, 40, 500011)
        res = tb.run_ac(tabs, d, tmax=2)  # far too few rounds for a 5 x 10 grid
        self.assertFalse(res["fp_conv"].any())  # no fixed point by round 2, and no run beyond it
        self.assertFalse(res["exact_fp"].any())  # no fixed point by T_max counts as wrong
        self.assertTrue((res["t_out"] <= 2).all())

    def test_fixed_point_clause_alone_halts_k4_oracle(self):
        tabs = om.build4()
        d = tb.fixed_set("m2d_dense", 8, 120, 500012)
        res = tb.run_ac(tabs, d)
        late = (res["Tmax"] - res["t_out"]) < res["MH"]  # window clause false
        self.assertTrue(late.any(), "test needs samples whose readout changes late")
        self.assertTrue(res["exact_halt"][late].all())  # halted only via the fixed-point clause
        self.assertTrue(res["fp_conv"].all())
        self.assertEqual(tb.summarize(res)["exact_halt"], 1.0)

    def test_window_clause_constant_readout(self):
        K = 2
        u = np.ones((8, K), np.uint8)
        r = np.ones((2, 8, 8, K, K), np.uint8)
        o = np.ones((8, 1 << K), np.uint8)
        d = tb.fixed_set("m2d_dense", 3, 20, 500013)
        d["y"] = np.ones_like(d["y"])
        res = tb.run_ac({"u": u, "r": r, "o": o}, d)
        self.assertTrue(res["exact_halt"].all())

    def test_exact_fp_equals_fp_readout_when_converged(self):
        tabs = om.build4()
        d = tb.fixed_set("m2d_last", 4, 40, 500014)
        res = tb.run_ac(tabs, d)
        self.assertTrue((res["exact_fp"] == (res["exact_T"] & res["fp_conv"])).all())


class TestFamilyAndNoTest(unittest.TestCase):
    def test_eval_family_exhaustive_matches_m1d_cert(self):
        rng = np.random.default_rng(500021)
        T = m1d_cert.random_tables(rng, 3)
        fam = tb.eval_family(T, 3, (8, 12), 50, 500022)
        for n in (8, 12):
            f = fam[str(n)]
            self.assertEqual(f["mode"], "exhaustive_fp")
            self.assertEqual(f["total"], 1 << n)
            self.assertEqual(f["correct"], m1d_cert.family_exact_match(T, n)["correct"])
            self.assertLessEqual(f["acc_fp"], f["bound_2^(K+2-n)"] + 1e-12)

    def test_eval_family_sampled_is_labelled(self):
        rng = np.random.default_rng(500023)
        T = m1d_cert.random_tables(rng, 2)
        fam = tb.eval_family(T, 2, (6,), 30, 500024, exhaustive_max_n=4)
        self.assertEqual(fam["6"]["mode"], "sampled_run_ac")
        self.assertNotIn("acc_fp", fam["6"])

    def test_no_test_skips_test_json(self):
        with tempfile.TemporaryDirectory() as td_:
            tb.main(["--task", "delay3", "--K", "5", "--seed", "300902", "--steps", "2", "--eval_every", "2",
                     "--n_dev", "10", "--no_test", "--out", td_])
            self.assertFalse(os.path.exists(os.path.join(td_, "test.json")))
            for f in ("config.json", "final.json", "tables.json", "log.json"):
                self.assertTrue(os.path.exists(os.path.join(td_, f)), f)
            with open(os.path.join(td_, "config.json"), encoding="utf-8") as f:
                self.assertTrue(json.load(f)["no_test"])

    def test_smoke_train_m1d_family_in_test_json(self):
        with tempfile.TemporaryDirectory() as td_:
            tb.main(["--task", "m1d", "--K", "2", "--seed", "300903", "--steps", "2", "--eval_every", "2",
                     "--n_dev", "10", "--n_test", "10", "--out", td_])
            with open(os.path.join(td_, "test.json"), encoding="utf-8") as f:
                t = json.load(f)
            self.assertEqual(t["family_a_2^(n-1)"]["16"]["mode"], "exhaustive_fp")


if __name__ == "__main__":
    unittest.main()
