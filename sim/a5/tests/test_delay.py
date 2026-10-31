import unittest

import numpy as np

from sim.a5 import ac, chain_utils as cu, delay

SEED = 200401


def rand_tables(rng, K, c=1, dens=0.7):
    nin = 1 << (c + 1)
    return dict(u=(rng.random((nin, K)) < 0.9).astype(np.uint8),
                r=(rng.random((1, nin, nin, K, K)) < dens).astype(np.uint8),
                o=(rng.random((nin, 1 << K)) < 0.5).astype(np.uint8))


class TestDelay(unittest.TestCase):
    def test_target(self):
        X, bits = delay.words(4)
        y = delay.delay_target(bits, 2)
        self.assertEqual(y[0b0110].tolist(), [0, 0, 0, 1])  # bits (LSB first) 0,1,1,0 -> y = 0,0,0,1
        self.assertTrue((y[:, :2] == 0).all())

    def test_construction_d1_to_4(self):
        for d in (1, 2, 3, 4):
            res = delay.verify_construction(d)
            self.assertEqual(res["wrong"], 0, res)
            self.assertEqual(res["K"], d + 2)

    def test_construction_vs_independent_interpreter_and_simulator(self):
        from sim.cert.tables import interpret
        rng = np.random.default_rng(SEED)
        for d in (2, 3):
            T = delay.build(d)
            tj = dict(dims=1, c=1, V=d + 2, u=T["u"].astype(int).tolist(), r=T["r"].astype(int).tolist(),
                      o=T["o"].astype(int).tolist())
            for _ in range(5):
                n = int(rng.integers(3, 10))
                X, y = delay.gen(rng, 1, n, d)
                _, pred, _ = interpret(tj, X[0])
                self.assertEqual(pred.tolist(), y[0].tolist())
                self.assertEqual(cu.ternary_predict(T, X[0]).tolist(), y[0].tolist())

    def test_construction_singleton_free_but_sets_encode_window(self):
        # the sets are NOT singletons (bit-vector code): check S_v = {star} + {i : x_{v-i}=1} on a sample word
        d = 3
        T = delay.build(d)
        bits = np.array([1, 0, 1, 1, 0, 1, 0])
        xi = (bits | 2)[None]
        c, _ = ac.fixed_point(T["u"], T["r"], xi)
        for v in range(len(bits)):
            want = {d + 1} | {i for i in range(d + 1) if v - i >= 0 and bits[v - i]}
            got = {h for h in range(d + 2) if c[0, v, h]}
            self.assertEqual(got, want)

    def test_pigeonhole_lower_bound(self):
        rng = np.random.default_rng(SEED + 1)
        for d, Ks in ((3, (1,)), (4, (1, 2)), (5, (1, 2, 3)), (6, (1, 2, 4))):
            for K in Ks:
                self.assertLessEqual(K, d - 2)
                for _ in range(5):
                    T = rand_tables(rng, K, dens=float(rng.choice([0.6, 0.8, 1.0])))
                    cert = delay.pigeonhole_counterexample(T, d)
                    self.assertIsNotNone(cert)
                    self.assertTrue(cert["confirmed_by_simulator"])
                    self.assertNotEqual(cert["pred"], cert["want"])

    def test_pigeonhole_on_the_construction_below_threshold(self):
        # the K=d+2 table built for d, used against a longer delay D=d+4: needs K <= D-2 -> must fail
        T = delay.build(2)
        cert = delay.pigeonhole_counterexample(T, 6)
        self.assertIsNotNone(cert)
        self.assertTrue(cert["confirmed_by_simulator"])
        # and the construction itself has NO counterexample against its own d (it is correct)
        self.assertIsNone(delay.pigeonhole_counterexample(delay.build(3), 3))

    def test_sat_instances_shape(self):
        inst = delay.sat_instances(2, 3)
        self.assertEqual(len(inst), 2 + 4 + 8)


if __name__ == "__main__":
    unittest.main()
