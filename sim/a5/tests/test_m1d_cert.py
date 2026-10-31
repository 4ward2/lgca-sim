import unittest

import numpy as np

from sim.a5 import ac, chain_utils as cu, delay, m1d_cert as mc, tasks_mul as tm

SEED = 200301


def and_table(K=2):
    """Hand-built: all pairs allowed, readout = a AND b of the cell itself (a plausible-looking but wrong 'multiplier')."""
    u = np.ones((8, K), np.uint8)
    r = np.ones((1, 8, 8, K, K), np.uint8)
    o = np.zeros((8, 1 << K), np.uint8)
    for x in range(8):
        o[x, :] = (x & 1) & ((x >> 1) & 1)
    return dict(u=u, r=r, o=o)


class TestM1D(unittest.TestCase):
    def test_words_match_task_generator(self):
        d = tm.generate_all("M1D", 3)
        # code word from m1d_cert equals ac.codes_from_x of the task generator
        xi = ac.codes_from_x(d["x"])
        for i, (a, b) in enumerate(zip(d["a"], d["b"])):
            self.assertTrue(np.array_equal(xi[i], mc.word(3, int(a), int(b))))
            self.assertTrue(np.array_equal(d["y"][i], mc.truth(3, int(a), int(b))))

    def test_left_scan_equals_fixed_point_chain_lemma(self):
        # S_v = L_v & R_v  =>  the leftmost cell's fixed-point set is <= L; the last-but-checked: L-scan is an upper bound
        rng = np.random.default_rng(SEED)
        for K in (2, 3, 4):
            T = mc.random_tables(rng, K)
            X, _ = mc.family_words(K + 2)
            L = cu.left_scan(T["u"], T["r"], X)
            c, _ = ac.fixed_point(T["u"], T["r"], X)
            S = ac.subset_index(c)
            self.assertTrue(((S & ~L) == 0).all())  # S subset of L (Lemma 1)

    def test_pigeonhole_random_tables(self):
        rng = np.random.default_rng(SEED + 1)
        for K in (2, 3, 4):
            for _ in range(6):
                T = mc.random_tables(rng, K, dens=float(rng.choice([0.5, 0.7, 0.9])))
                cert = mc.pigeonhole_counterexample(T)
                self.assertEqual(cert["n"], K + 3)
                self.assertTrue(cert["wrong"])
                self.assertTrue(cert["simulator_agrees"])
                # independent confirmation: re-simulate and compare to integer arithmetic
                w = np.array(cert["word"])
                pred = cu.predict(T, w[None])[0]
                self.assertTrue(np.array_equal(pred, cert["pred"]))
                self.assertFalse(np.array_equal(pred, mc.truth(cert["n"], cert["a"], cert["b"])))

    def test_pigeonhole_hand_tables(self):
        tabs = [and_table(2), and_table(3)]
        for d in (1, 2):  # delay-line tables used as multiplier candidates (K = d+2)
            tabs.append(delay.build(d, c=2))
        # the all-allow table with o constant
        allow = and_table(2)
        allow["o"][:] = 1
        tabs.append(allow)
        for T in tabs:
            cert = mc.pigeonhole_counterexample(T)
            self.assertTrue(cert["wrong"] and cert["simulator_agrees"])
            self.assertLessEqual(cert["n"], T["u"].shape[1] + 3)
            s = mc.smallest_counterexample(T)
            self.assertLessEqual(s["n"], T["u"].shape[1] + 3)
            self.assertTrue(s["wrong"] and s["simulator_agrees"])

    def test_accuracy_bound(self):
        rng = np.random.default_rng(SEED + 2)
        for K, ns in ((2, (5, 6, 8, 10)), (3, (6, 8, 10)), (4, (7, 8, 10, 11))):
            for _ in range(4):
                T = mc.random_tables(rng, K)
                for n in ns:
                    res = mc.check_theorem2(T, n)
                    self.assertLessEqual(res["correct"], 2 ** (K + 2))
        for T in (and_table(2), delay.build(2, c=2)):
            K = T["u"].shape[1]
            for n in range(K + 3, K + 8):
                mc.check_theorem2(T, n)

    def test_bound_numbers(self):
        self.assertEqual(mc.family_bound(2, 8), 2.0 ** -4)
        self.assertEqual(mc.family_bound(4, 8), 0.25)
        self.assertAlmostEqual(mc.family_bound(2, 12), 2.0 ** -8)


if __name__ == "__main__":
    unittest.main()
