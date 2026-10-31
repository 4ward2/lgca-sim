import unittest

import numpy as np
import torch

from sim.a5 import ac, oracle_m2d as om, tasks_mul as tm  # importing sim.a5 puts sim/ on sys.path

SEED = 200010


class TestInterpreterVsTernaryNet(unittest.TestCase):
    def _check(self, dims, shape):
        import b_arm as B
        from logic import to_internal
        rng = np.random.default_rng(SEED + dims)
        for trial in range(6):
            net = B.TernaryNet(dims, 2, V=2)
            dens = [0.5, 0.7, 0.85][trial % 3]
            u = (rng.random(tuple(net.u.shape)) < dens).astype(np.float32)
            r = (rng.random(tuple(net.r.shape)) < dens).astype(np.float32)
            o = (rng.random(tuple(net.o.shape)) < 0.5).astype(np.float32)
            with torch.no_grad():
                net.u.copy_(torch.as_tensor(u))
                net.r.copy_(torch.as_tensor(r))
                net.o.copy_(torch.as_tensor(o))
            x = (rng.random((5, 2) + shape) < 0.5).astype(np.float32)
            xv = to_internal(x)
            xi_t = B.xidx(xv)
            xi = ac.codes_from_x(x)
            self.assertTrue(np.array_equal(xi_t.numpy(), xi))
            ct = xv.new_ones(tuple(xv.shape[1:]) + (2,))
            cn = np.ones(xi.shape + (2,), bool)
            for _ in range(12):  # compare every synchronous step, not just the fixed point
                ct = net.update(ct, xi_t)
                cn = ac.update(u, r, cn, xi)
                self.assertTrue(np.array_equal(ct.detach().numpy().astype(bool), cn))
            self.assertTrue(np.array_equal(net.readout(ct, xi_t).detach().numpy(), ac.readout(o, cn, xi)))

    def test_2d(self):
        self._check(2, (3, 4))

    def test_1d(self):
        self._check(1, (7,))


class TestOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T = om.build()

    def _verify(self, d):
        y, c, steps = om.predict(self.T, d["x"])
        self.assertTrue(np.array_equal(y, d["y"].astype(np.uint8)), "per-cell readout wrong")
        self.assertTrue((c.sum(-1) == 1).all(), "AC did not reach singleton domains")
        return steps

    def test_exhaustive_n_le_4(self):
        for n in (1, 2, 3, 4):
            self._verify(tm.generate_all("M2D", n))

    def test_random_n_5_to_6(self):
        rng = np.random.default_rng(SEED + 10)
        for n in (5, 6):
            self._verify(tm.generate("M2D", rng, 300, n))

    def test_steps_linear(self):
        # a wave from the top-left corner, not a cyclic fixed point: steps = 3n - 1
        steps = [self._verify(tm.generate_all("M2D", n)) for n in (2, 3, 4)]
        self.assertEqual(steps, [5, 8, 11])


if __name__ == "__main__":
    unittest.main()
