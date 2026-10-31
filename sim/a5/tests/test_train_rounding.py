import unittest

import numpy as np
import torch

from sim.a5 import rounding_k as rk, tasks_delay as td, tasks_mul as tm

SEED = 300001


class TestRoundingK2(unittest.TestCase):
    def _agree(self, dims, seeds):
        import b_arm as B
        for s in seeds:
            torch.manual_seed(SEED + s)
            c = 2
            bp = B.BPNet(dims, c, K=2, seed=SEED + s, init=1.5)  # large init so that the thresholds are exercised
            rng = np.random.default_rng(SEED + 100 + s)
            if dims == 1:
                x = (rng.random((40, c, 11)) < 0.5).astype(np.float32)
            else:
                x = (rng.random((20, c, 4, 5)) < 0.5).astype(np.float32)
            y = (rng.random(x.shape[:1] + x.shape[2:]) < 0.5).astype(np.float32)
            mask = rng.random(y.shape) < 0.8
            fit = dict(x=x, y=y, mask=mask)
            tmax = 4 * x.shape[-1] if dims == 1 else x.shape[-1] ** 2  # what bp_to_ternary uses
            ref = B.bp_to_ternary(bp, fit)
            mine, _ = rk.round_bp(bp, fit, tmax)
            self.assertTrue(np.array_equal(ref.u.detach().numpy().astype(np.uint8), mine["u"]))
            self.assertTrue(np.array_equal(ref.r.detach().numpy().astype(np.uint8), mine["r"]))
            self.assertTrue(np.array_equal(ref.o.detach().numpy().astype(np.uint8), mine["o"]))
            self.assertTrue(mine["o"].any())

    def test_1d(self):
        self._agree(1, range(5))

    def test_2d(self):
        self._agree(2, range(4))

    def test_general_k_shapes_and_default_zero(self):
        import b_arm as B
        for K in (3, 4):
            bp = B.BPNet(1, 2, K=K, seed=SEED + 7, init=1.0)
            rng = np.random.default_rng(SEED + 8)
            d = tm.gen_m1d(rng, 16, 3)
            tabs, nseen = rk.round_bp(bp, d, rk.default_tmax(d["x"].shape))
            self.assertEqual(tabs["o"].shape, (bp.nin, 1 << K))
            self.assertEqual(tabs["r"].shape, (1, bp.nin, bp.nin, K, K))
            pop = np.array([bin(i).count("1") for i in range(1 << K)])
            # a 1 in a non-singleton column can only come from an observed (x, S) pair
            self.assertLessEqual(int((tabs["o"][:, pop != 1] > 0).sum()), nseen)

    def test_json_roundtrip(self):
        import b_arm as B
        bp = B.BPNet(1, 1, K=4, seed=SEED + 9, init=1.0)
        rng = np.random.default_rng(SEED + 10)
        d = td.generate(rng, 16, 9, 3)
        tabs, _ = rk.round_bp(bp, d, 36)
        back = rk.from_json(rk.to_json(tabs))
        for k in "uro":
            self.assertTrue(np.array_equal(tabs[k], back[k]))


class TestDelay(unittest.TestCase):
    def test_reference(self):
        rng = np.random.default_rng(SEED + 20)
        for d in (1, 3, 4):
            g = td.generate(rng, 8, 12, d)
            for i in range(8):
                self.assertEqual(g["y"][i].astype(int).tolist(), td.ref_delay(g["x"][i, 0].astype(int).tolist(), d))
            self.assertTrue((g["y"][:, :d] == 0).all())
            self.assertEqual(g["x"].shape, (8, 1, 12))


if __name__ == "__main__":
    unittest.main()
