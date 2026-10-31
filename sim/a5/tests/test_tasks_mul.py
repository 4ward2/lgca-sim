import unittest

import numpy as np

from sim.a5 import tasks_mul as tm

SEED = 200001


class TestTasksMul(unittest.TestCase):
    def test_m2d_reference_matches_integer_product_and_local_rule(self):
        rng = np.random.default_rng(SEED)
        for n in (1, 2, 3, 5, 8, 12):
            for _ in range(60):
                a, b = int(rng.integers(0, 1 << n)), int(rng.integers(0, 1 << n))
                x, s, c = tm.ref_m2d(a, b, n)
                self.assertEqual(sum(v << k for k, v in enumerate(s[-1])), a * b)
                for j in range(n):  # running sum after row j is a*(b mod 2^(j+1))
                    self.assertEqual(sum(v << k for k, v in enumerate(s[j])), a * (b % (1 << (j + 1))))
                s2, c2 = tm.local_rule_m2d(x)
                self.assertEqual(s, s2)
                self.assertEqual(c, c2)

    def test_exhaustive_small_n(self):
        for n in (1, 2, 3, 4):
            for task in ("M1D", "M2D"):
                d = tm.generate_all(task, n)
                self.assertEqual(len(d["x"]), 4 ** n)
                for i in range(len(d["x"])):
                    a, b = int(d["a"][i]), int(d["b"][i])
                    out = d["y"][i][-1] if task == "M2D" else d["y"][i]
                    self.assertEqual(sum(int(v) << k for k, v in enumerate(out)), a * b)
                if task == "M2D":
                    for i in range(0, len(d["x"]), 7 if n == 4 else 1):
                        s, c = tm.local_rule_m2d(d["x"][i].astype(int).tolist())
                        self.assertEqual(s, d["y"][i].astype(int).tolist())

    def test_layouts(self):
        d = tm.generate_all("M1D", 3)
        i = int(np.nonzero((d["a"] == 5) & (d["b"] == 6))[0][0])
        self.assertEqual(d["x"].shape, (64, 2, 6))
        self.assertEqual(d["x"][i, 0].tolist(), [1, 0, 1, 0, 0, 0])  # a = 5 LSB first, padding zeros
        self.assertEqual(d["x"][i, 1].tolist(), [0, 1, 1, 0, 0, 0])  # b = 6
        self.assertEqual(d["y"][i].tolist(), [0, 1, 1, 1, 1, 0])  # 30 = 011110b
        d = tm.generate_all("M2D", 3)
        i = int(np.nonzero((d["a"] == 5) & (d["b"] == 6))[0][0])
        self.assertEqual(d["x"].shape, (64, 2, 3, 6))
        self.assertEqual(d["x"][i, 0, 1].tolist(), [1] * 6)                 # b_1 broadcast on row 1
        self.assertEqual(d["x"][i, 1, 1].tolist(), [0, 1, 0, 1, 0, 0])       # a shifted by 1
        self.assertTrue(d["mask"][:, -1].all() and not d["mask"][:, :-1].any())
        self.assertTrue(d["mask_all"].all())

    def test_random_generators(self):
        rng = np.random.default_rng(SEED + 1)
        for task in ("M1D", "M2D"):
            d = tm.generate(task, rng, 40, 9)
            self.assertEqual(d["diff"].shape, (40,))
            for i in range(40):
                out = d["y"][i][-1] if task == "M2D" else d["y"][i]
                self.assertEqual(sum(int(v) << k for k, v in enumerate(out)), int(d["a"][i]) * int(d["b"][i]))


if __name__ == "__main__":
    unittest.main()
