import unittest

import numpy as np

from sim.a5 import k2_search as ks


class TestK2Search(unittest.TestCase):
    def test_size_report(self):
        P = ks.problem(2)
        s = ks.space_size(P)
        self.assertEqual(s["inputs"], 16)
        self.assertEqual(s["u_bits"], 8)
        self.assertEqual(s["r_bits"], 4 * s["pairs_read"])
        self.assertGreater(s["log2_space_searched"], 100)

    def test_conflict_and_score_machinery(self):
        # all-ones tables: every set is {0,1}; scored cells share a key -> targets conflict, score > 0
        P = ks.problem(2)
        u = np.ones((8, 2), np.uint8)
        r = np.ones((2, 8, 8, 2, 2), np.uint8)
        sets = ks.final_sets(P, u, r)
        self.assertTrue((sets == 3).all())
        self.assertTrue(ks.conflict(P, sets, sets))
        self.assertGreater(ks.misclassified(P, sets), 0)

    def test_monotone_bounds(self):
        # S_lo subset of S_hi for partial assignments (soundness of the pruning bound)
        P = ks.problem(2)
        rng = np.random.default_rng(200021)
        for _ in range(5):
            val = {v: int(rng.integers(0, 2)) for v in P["vars"] if rng.random() < 0.6}
            (ulo, rlo), (uhi, rhi) = ks.make_tables(val)
            lo, hi = ks.final_sets(P, ulo, rlo), ks.final_sets(P, uhi, rhi)
            self.assertTrue(((lo & ~hi) == 0).all())

    def test_alive_never_prunes_prefix_of_correct_table(self):
        # soundness: every prefix of a complete correct (conflict-free, symmetry-canonical) table must stay alive
        P = ks.problem(1)
        rng = np.random.default_rng(200022)
        found = 0
        for _ in range(300):
            val = {v: int(rng.integers(0, 2)) for v in P["vars"]}
            (u, r), _ = ks.make_tables(val)
            if ks.misclassified(P, ks.final_sets(P, u, r)) == 0 and ks.sym_ok(P, val):
                found += 1
                for i in range(len(P["vars"]) + 1):
                    self.assertTrue(ks.alive(P, {v: val[v] for v in P["vars"][:i]}))
        self.assertGreater(found, 0)

    def test_cli_short(self):
        res = ks.main(["dfs", "--n", "1", "--time-limit", "20", "--split-bits", "4"])
        self.assertIn("dfs", res)


if __name__ == "__main__":
    unittest.main()
