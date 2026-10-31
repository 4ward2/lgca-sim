import json
import math
import os
import shutil
import tempfile
import unittest

from sim.a5 import dag_check, make_jobs_a5 as mj, report_a5, train_r_probe as rp

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(os.path.dirname(HERE), "jobs_a5.json")


class TestSchedule(unittest.TestCase):
    def test_wd_hits_target_decay(self):
        for head, tail in (("cos", 0.1), ("const", 1.0)):
            for x in (1, 3, 10, 30):
                wd = rp.wd_for_decay(x, 0.1, 60000, 6000, head, tail)
                self.assertAlmostEqual(rp.total_decay(wd, 0.1, 60000, 6000, head, tail), x, places=6)

    def test_multiplier_shape(self):
        self.assertAlmostEqual(rp.lr_mult(0, 6000, "cos", 0.1), 1.0)
        self.assertAlmostEqual(rp.lr_mult(3000, 6000, "cos", 0.1), 0.55)
        self.assertEqual(rp.lr_mult(6000, 6000, "cos", 0.1), 0.1)
        self.assertEqual(rp.lr_mult(59999, 6000, "cos", 0.1), 0.1)
        self.assertEqual(rp.lr_mult(100, 4000, "const", 1.0), 1.0)

    def test_criterion(self):
        mk = lambda s, h16, h32: dict(step=s, hard_16=h16, hard_32=h32, hard_64=0.0, soft_32=0.0)
        log = [mk(500 * i, 1.0, 0.0) for i in range(1, 30)]
        r = rp.summarize_log(log)
        self.assertTrue(r["train_sustained_reached"])
        self.assertFalse(r["flipped"])
        log += [mk(500 * i, 1.0, 0.95) for i in range(30, 32)]
        self.assertTrue(rp.summarize_log(log)["flipped"])
        # hard_16 collapsed before a later hard_32 > 0.9: not a flip (needs hard_16 >= 0.99 at that same step)
        col = [mk(500 * i, 1.0, 0.0) for i in range(1, 30)] + [mk(500 * i, 0.2, 0.95) for i in range(30, 40)]
        rc = rp.summarize_log(col)
        self.assertTrue(rc["train_sustained_reached"])
        self.assertFalse(rc["flipped"])
        self.assertTrue(rp.summarize_log(col + [mk(500 * 40, 1.0, 0.95)])["flipped"])  # both at the same later step
        early = [mk(500, 1.0, 0.95)] + [mk(500 * i, 1.0, 0.0) for i in range(2, 40)]  # hit before sustained_from
        self.assertFalse(rp.summarize_log(early)["flipped"])
        bad = [mk(500 * i, 1.0 if i % 10 else 0.0, 0.99) for i in range(1, 40)]
        self.assertFalse(rp.summarize_log(bad)["train_sustained_reached"])


class TestResume(unittest.TestCase):
    def test_resume_matches_straight_run(self):
        d = tempfile.mkdtemp()
        try:
            common = ["--task", "P", "--seed", "300301", "--decay", "3", "--steps", "6", "--eval_every", "3",
                      "--lengths", "16", "--head_steps", "4"]
            rp.main(common + ["--out", os.path.join(d, "full")])
            rp.main(common + ["--stop_at", "3", "--out", os.path.join(d, "s1")])
            rp.main(common + ["--resume", os.path.join(d, "s1"), "--out", os.path.join(d, "s2")])
            read = lambda p: [json.loads(x) for x in open(os.path.join(d, p, "log.jsonl"), encoding="utf-8")]
            a, b = read("full"), read("s2")
            self.assertEqual([r["step"] for r in a], [r["step"] for r in b])
            for ra, rb in zip(a, b):
                self.assertAlmostEqual(ra["train_loss_ema"], rb["train_loss_ema"], places=6)
                self.assertAlmostEqual(ra["param_norm"], rb["param_norm"], places=4)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestDag(unittest.TestCase):
    def test_committed_json_matches_generator(self):
        self.assertEqual(json.load(open(JOBS, encoding="utf-8")), json.loads(json.dumps(mj.build())))

    def test_valid(self):
        errs, n = dag_check.check(JOBS)
        self.assertEqual(errs, [])
        self.assertGreater(n, 100)

    def test_sat_jobs_pilot_flags_and_verdict(self):
        jobs = {j["name"]: j for j in mj.build()["jobs"]}
        sat = [j for j in jobs.values() if j.get("kind") == "SAT"]
        self.assertEqual(len(sat), 12)
        for j in sat:
            self.assertTrue(j["command"].startswith("pip install -q python-sat &&"), j["name"])
            self.assertIn("--time-limit 14400", j["command"])
            self.assertIn(f"--out out/{j['name']}/k.cnf", j["command"])
        for j in jobs.values():
            if j.get("kind") == "B":
                self.assertEqual("--no_test" in j["command"], not j["formal"], j["name"])
        v = jobs["verdict"]
        self.assertTrue({"report"} | {j["name"] for j in sat} <= set(v["needs"]))
        self.assertEqual(sum(1 for n in v["needs"] if n.endswith("-formal")), 15)
        self.assertEqual(sum(1 for n in v["needs"] if n.endswith("-g4")), 30)
        self.assertEqual(len(jobs), 164)

    def test_detects_bad_dag(self):
        d = mj.build()
        d["jobs"][0]["command"] += " && cp x out/other/"
        d["jobs"][3]["needs"].append("nope")
        p = os.path.join(tempfile.mkdtemp(), "j.json")
        json.dump(d, open(p, "w"))
        errs, _ = dag_check.check(p)
        self.assertGreaterEqual(len(errs), 2)


if __name__ == "__main__":
    unittest.main()
