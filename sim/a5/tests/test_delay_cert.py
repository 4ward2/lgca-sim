import unittest

import numpy as np

from sim.a5 import delay, delay_cert as dc

SEED = 500101


def rand_tables(rng, K, nin=4, dens=0.7):
    return dict(u=(rng.random((nin, K)) < 0.9).astype(np.uint8),
                r=(rng.random((1, nin, nin, K, K)) < dens).astype(np.uint8),
                o=(rng.random((nin, 1 << K)) < 0.5).astype(np.uint8))


def copy(T):
    return {k: v.copy() for k, v in T.items()}


class TestDelayCert(unittest.TestCase):
    def test_build_passes(self):
        for d in (1, 2, 3, 4):
            res = dc.certify(delay.build(d), d)
            self.assertEqual(res["status"], "PASS", (d, res))

    def test_corrupted_readout_fails_with_confirmed_cex(self):
        for d in (1, 2, 3, 4):
            T = copy(delay.build(d))
            T["o"][delay.code_of(0), (1 << (d + 1)) | (1 << d)] ^= 1  # reachable (x=0, S={d, star})
            res = dc.certify(T, d)
            self.assertEqual(res["status"], "FAIL", (d, res))
            self.assertTrue(res["interpreter_confirms"] and res["simulator_agrees"])
            self.assertNotEqual(res["pred"], res["want"])

    def test_corrupted_relation_fails(self):
        for d in (2, 3):
            T = copy(delay.build(d))
            T["r"][0, 3, 3, 0, 1] = 0   # break shift 0->1 after a 1 followed by a 1
            res = dc.certify(T, d)
            self.assertEqual(res["status"], "FAIL", (d, res))
            self.assertTrue(res["interpreter_confirms"] and res["simulator_agrees"])

    def test_wrong_delay_fails(self):
        res = dc.certify(delay.build(3), 4)
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(res["interpreter_confirms"])

    def test_random_tables_fail(self):
        rng = np.random.default_rng(SEED)
        for K in (3, 5, 8):
            res = dc.certify(rand_tables(rng, K), 3)
            self.assertEqual(res["status"], "FAIL", K)
            self.assertTrue(res["interpreter_confirms"] and res["simulator_agrees"])

    def test_pass_agrees_with_exhaustive_words(self):
        # a PASS table is exact on every word (small lengths exhaustively)
        from sim.a5 import chain_utils as cu
        d = 3
        T = delay.build(d)
        for n in range(1, 9):
            X, bits = delay.words(n)
            self.assertTrue((cu.predict(T, X) == delay.delay_target(bits, d)).all())


if __name__ == "__main__":
    unittest.main()
