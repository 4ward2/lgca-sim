import unittest

import numpy as np

from sim.a5 import oracle_m2d as om, tasks_mul as tm, wave_cert as wc

SEED = 200601


def copy(T):
    return {k: v.copy() for k, v in T.items()}


def relabel(T, perm):
    """Rename latent values h -> perm[h] in all tables (a symmetry: the certificate must still PASS)."""
    K = len(perm)
    inv = np.argsort(perm)
    u = T["u"][:, inv]
    r = T["r"][:, :, :, inv][:, :, :, :, inv]
    o = np.zeros_like(T["o"])
    for S in range(1 << K):
        S2 = sum(1 << perm[h] for h in range(K) if (S >> h) & 1)
        o[:, S2] = T["o"][:, S]
    return dict(u=u, r=r, o=o)


def exhaustively_correct(T, n_max=4):
    for n in range(1, n_max + 1):
        d = tm.generate_all("M2D", n)
        y, _, _ = wc._predict(T, d["x"])
        if not np.array_equal(y, d["y"].astype(np.uint8)):
            return False
    return True


class TestWaveCert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T4 = om.build4()
        cls.T16 = om.build()

    def test_oracles_pass(self):
        for T in (self.T4, self.T16):
            res = wc.certify(T)
            self.assertEqual(res["status"], "PASS", res)
            self.assertEqual(res["rounds_bound"], "3n-1")

    def test_pass_is_relabelling_invariant(self):
        rng = np.random.default_rng(SEED)
        for _ in range(3):
            perm = rng.permutation(4)
            self.assertEqual(wc.certify(relabel(self.T4, perm))["status"], "PASS")
        self.assertTrue(exhaustively_correct(relabel(self.T4, np.array([2, 0, 3, 1]))))

    def test_deliberate_corruptions_fail(self):
        cases = {}
        T = copy(self.T4)                      # drop top-row anchoring
        T["r"][0, 0, 4:] = 1
        cases["no top anchor"] = T
        T = copy(self.T4)                      # drop left-column anchoring
        T["r"][1, 0, 4:] = 1
        cases["no left anchor"] = T
        T = copy(self.T4)                      # wrong readout on one singleton
        T["o"][5, 1 << 2] ^= 1
        cases["bad readout"] = T
        T = copy(self.T4)                      # wrong vertical copy relation
        T["r"][0, 6, 7, 1, 2] ^= 1
        cases["bad vertical relation"] = T
        T = copy(self.T16)
        T["u"][7, 3] ^= 1
        cases["K16 bad unary"] = T
        T = copy(self.T16)
        T["r"][1, 0, 5:] = 1
        cases["K16 no left anchor"] = T
        for name, T in cases.items():
            res = wc.certify(T)
            self.assertNotEqual(res["status"], "PASS", name)
            if res["status"] == "FAIL":
                cex = res["counterexample"]
                self.assertTrue(cex["simulator_agrees"], name)
                self.assertNotEqual(cex["pred"], cex["want"], name)
        # these particular corruptions are all genuinely wrong -> FAIL (not merely UNKNOWN)
        for name in ("no top anchor", "no left anchor", "bad readout"):
            self.assertEqual(wc.certify(cases[name])["status"], "FAIL", name)

    def test_soundness_random_single_bit_flips(self):
        rng = np.random.default_rng(SEED + 1)
        counts = {"PASS": 0, "FAIL": 0, "UNKNOWN": 0}
        for i in range(60):
            T = copy(self.T4)
            kind = ["u", "r", "o"][i % 3]
            idx = tuple(int(rng.integers(0, s)) for s in T[kind].shape)
            T[kind][idx] ^= 1
            res = wc.certify(T)
            counts[res["status"]] += 1
            if res["status"] == "PASS":      # PASS must imply true exhaustive correctness (and rounds bound)
                self.assertTrue(exhaustively_correct(T), (kind, idx))
                d = tm.generate_all("M2D", 3)
                self.assertLessEqual(int(om.strict_change_rounds(T, d["x"]).max()), 3 * 3 - 1)
            if res["status"] == "FAIL":
                self.assertTrue(res["counterexample"]["simulator_agrees"])
        print("random single-bit flips of the K=4 oracle:", counts)
        self.assertGreater(counts["FAIL"], 0)

    def test_wave_check_reports_reason(self):
        T = copy(self.T4)
        T["u"][4:, :] = 1
        T["r"][0, 0, 4:] = 1
        T["r"][1, 0, 4:] = 1
        w = wc.wave_check(T)
        self.assertFalse(w["ok"])
        self.assertIn("singleton", w["reason"])


if __name__ == "__main__":
    unittest.main()
