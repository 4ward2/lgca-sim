import json
import os
import tempfile
import unittest

import numpy as np

from sim.a5 import oracle_m2d as om, train_b as tb


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


if __name__ == "__main__":
    unittest.main()
