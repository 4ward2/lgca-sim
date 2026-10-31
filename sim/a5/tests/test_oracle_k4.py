import unittest

import numpy as np

from sim.a5 import ac, chain_utils as cu, oracle_m2d as om, tasks_mul as tm

SEED = 200501


class TestK4Oracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T4 = om.build4()
        cls.T16 = om.build()

    def _verify(self, T, d, bound_rounds=True):
        y, c, steps = om.predict(T, d["x"])
        self.assertTrue(np.array_equal(y, d["y"].astype(np.uint8)), "per-cell (dense) readout wrong")
        self.assertTrue((c.sum(-1) == 1).all(), "not all cells singleton")
        return steps

    def test_k4_exhaustive_n_le_4_all_rows(self):
        for n in (1, 2, 3, 4):
            d = tm.generate_all("M2D", n)
            self._verify(self.T4, d)
            rounds = om.strict_change_rounds(self.T4, d["x"])
            self.assertLessEqual(int(rounds.max()), 3 * n)
            self.assertEqual(int(rounds.max()), 3 * n - 1 if n > 1 else rounds.max())

    def test_k4_random_n_5_to_8(self):
        rng = np.random.default_rng(SEED)
        seen = {}
        for n in (5, 6, 7, 8):
            d = tm.generate(rng=rng, task="M2D", B=60, n=n)
            self._verify(self.T4, d)
            rounds = om.strict_change_rounds(self.T4, d["x"])
            self.assertLessEqual(int(rounds.max()), 3 * n - 1)
            seen[n] = int(rounds.max())
        print('K=4 max strict-change rounds (random samples):', seen)

    def test_k16_rounds_bound_and_same_function(self):
        for n in (1, 2, 3, 4):
            d = tm.generate_all("M2D", n)
            self._verify(self.T16, d)
            self.assertLessEqual(int(om.strict_change_rounds(self.T16, d["x"]).max()), 3 * n)
        rng = np.random.default_rng(SEED + 1)
        d = tm.generate("M2D", rng, 40, 7)
        self._verify(self.T16, d)

    def test_k4_matches_ternarynet_simulator(self):
        d = tm.generate_all("M2D", 3)
        xi = ac.codes_from_x(d["x"])
        for i in (0, 17, 40, 63):
            tern = cu.ternary_predict(self.T4, xi[i])
            self.assertTrue(np.array_equal(tern, d["y"][i].astype(np.int64)))

    def test_anchoring_explicit(self):
        for T in (self.T4, self.T16):
            self.assertTrue(all(om.check_anchoring(T).values()), om.check_anchoring(T))

    def test_anchoring_is_necessary(self):
        # remove the top / left anchoring: every cell keeps a larger set and the readout is wrong (n=2, 3)
        for kind in ("top", "left", "both"):
            T = {k: v.copy() for k, v in self.T4.items()}
            for xb in range(8):
                if (xb >> 2) & 1:
                    if kind in ("top", "both"):
                        T["r"][0, 0, xb] = 1
                    if kind in ("left", "both"):
                        T["r"][1, 0, xb] = 1
            self.assertFalse(all(om.check_anchoring(T).values()))
            bad = 0
            d = tm.generate_all("M2D", 3)
            y, c, _ = om.predict(T, d["x"])
            bad = int((y != d["y"].astype(np.uint8)).any(axis=(1, 2)).sum())
            self.assertGreater(bad, 0, kind)
            self.assertTrue((c.sum(-1) > 1).any())

    def test_k4_tables_shapes_and_nonsingleton_readout_zero(self):
        T = self.T4
        self.assertEqual(T["u"].shape, (8, 4))
        self.assertEqual(T["r"].shape, (2, 8, 8, 4, 4))
        self.assertEqual(T["o"].shape, (8, 16))
        for S in range(16):
            if bin(S).count("1") != 1:
                self.assertTrue((T["o"][:, S] == 0).all())


if __name__ == "__main__":
    unittest.main()
