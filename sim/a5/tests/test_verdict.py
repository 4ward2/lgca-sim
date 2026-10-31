import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from sim.a5 import delay, make_jobs_a5 as mj, oracle_m2d as om, rounding_k as rk, train_b as tb, verdict_a5 as va

SEEDS = [500201, 500202]
EX = dict(
    b_configs=[("m1d", 2), ("delay3", 1), ("delay4", 2), ("delay3", 5), ("m2d_dense", 4), ("m2d_dense", 2), ("m2d_last", 2)],
    seeds=SEEDS, r_runs=[("P", 1, 0), ("P", 3, 0), ("A", 500301, 0)], segments=4, sat=["sat-m2d-n2-k4-last", "sat-delay-d2-k2"])


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def rand_tables(rng, K, nin):
    return dict(u=(rng.random((nin, K)) < 0.9).astype(np.uint8),
                r=(rng.random((1, nin, nin, K, K)) < 0.7).astype(np.uint8),
                o=(rng.random((nin, 1 << K)) < 0.5).astype(np.uint8))


def rand_m2d(rng, K):
    return dict(u=(rng.random((8, K)) < 0.9).astype(np.uint8), r=(rng.random((2, 8, 8, K, K)) < 0.7).astype(np.uint8),
                o=(rng.random((8, 1 << K)) < 0.5).astype(np.uint8))


def write_b(root, task, K, tabs, halt=1.0, seeds=SEEDS, test=True):
    for s in seeds:
        d = os.path.join(root, f"b-{task}-k{K}-formal", f"s{s}")
        dump(rk.to_json(tabs) | dict(K=K), os.path.join(d, "tables.json"))
        dump(dict(dev_score=1.0, best_deploy_step=2000), os.path.join(d, "final.json"))
        if test:
            dump(dict(task=task, K=K, seed=s, per_size={str(n): dict(exact_halt=halt, exact_fp=halt)
                                                         for n in tb.TASKS[task]["test"]}), os.path.join(d, "test.json"))


def write_r(root, task, s, flip, complete=True, collapse=False):
    name = f"r{task}-d0-s{s}-g4"
    log = [dict(step=500 * i, hard_16=1.0, hard_32=0.0, hard_64=0.0, soft_32=0.0) for i in range(1, 60)]
    if flip:
        for r in log[40:]:
            r["hard_32"] = 0.95
    if collapse:
        for r in log[40:]:
            r["hard_32"], r["hard_16"] = 0.95, 0.1
    os.makedirs(os.path.join(root, name), exist_ok=True)
    with open(os.path.join(root, name, "log.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in log))
    dump(dict(task=task, seed=s, wd=0.0, complete=complete), os.path.join(root, name, "summary.json"))


def full_root(root, halt_m2d4=1.0):
    rng = np.random.default_rng(500101)
    write_b(root, "m1d", 2, rand_tables(rng, 2, 8), halt=0.0)
    write_b(root, "delay3", 1, rand_tables(rng, 1, 4), halt=0.0)
    write_b(root, "delay4", 2, rand_tables(rng, 2, 4), halt=0.0)
    write_b(root, "delay3", 5, delay.build(3), halt=1.0)
    write_b(root, "m2d_dense", 4, om.build4(), halt=halt_m2d4)
    write_b(root, "m2d_dense", 2, rand_m2d(rng, 2), halt=0.0)
    write_b(root, "m2d_last", 2, rand_m2d(rng, 2), halt=0.0)
    write_r(root, "P", 1, flip=True)
    write_r(root, "P", 3, flip=False)
    write_r(root, "A", 500301, flip=False)
    dump(dict(status="UNSAT", task="m2d", n=2, K=4, d=None, targets="last", vars=1, clauses=1, seconds=1.0),
         os.path.join(root, "sat-m2d-n2-k4-last", "k.cnf.result.json"))
    dump(dict(status="UNSAT", task="delay", n=5, K=2, d=2, targets="all", vars=1, clauses=1, seconds=1.0),
         os.path.join(root, "sat-delay-d2-k2", "k.cnf.result.json"))


class TestVerdict(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_full_synthetic_run(self):
        full_root(self.root)
        v = va.compute(self.root, EX)
        self.assertEqual(v["P1"]["status"], "holds", v["P1"])
        for t in v["P1"]["per_table"].values():
            self.assertEqual(set(t["family"]), {"8", "12", "16"})
            self.assertTrue(t["counterexample"]["simulator_agrees"] and t["counterexample"]["n"] <= 2 + 3)
        self.assertEqual(v["P2"]["status"], "holds", v["P2"])
        self.assertEqual(len(v["P2"]["per_table"]), 4)
        p3 = v["P3"]["per_config"]
        self.assertEqual(p3["m2d_dense-K4"]["status"], "learnable")
        self.assertEqual(p3["m2d_dense-K4"]["n_cert_pass"], 2)
        self.assertEqual(p3["delay3-K5"]["status"], "learnable")
        self.assertEqual(p3["delay3-K5"]["n_cert_pass"], 2)
        self.assertIn("nonsingleton_frac_mean", p3["m2d_dense-K4"])
        self.assertEqual(v["C"]["status"], "supported")
        self.assertTrue(v["P4"]["status"].startswith("prediction holds"), v["P4"])
        self.assertEqual(v["P4"]["sat"]["sat-m2d-n2-k4-last"]["status"], "UNSAT")
        self.assertIn("control", v["P4"]["notes"][0])  # K=4 control not SAT is flagged
        self.assertEqual(v["R1"]["status"], "detection power shown")
        self.assertTrue(v["R2"]["status"].startswith("not observed"))
        md = va.to_markdown(v)
        self.assertIn("P1", md)

    def test_not_learnable_and_k2_refuted(self):
        full_root(self.root, halt_m2d4=0.5)
        write_b(self.root, "m2d_last", 2, rand_m2d(np.random.default_rng(500102), 2), halt=0.97)
        v = va.compute(self.root, EX)
        self.assertEqual(v["P3"]["per_config"]["m2d_dense-K4"]["status"], "not learnable")
        self.assertEqual(v["C"]["status"], "not supported")
        self.assertTrue(v["P4"]["status"].startswith("prediction refuted"))

    def test_certificate_fail_blocks_learnable(self):
        full_root(self.root)
        T = om.build4()
        T["o"][:, :] = 0  # wrong readout: certificate fails although the reported exact_halt says 1.0
        write_b(self.root, "m2d_dense", 4, T, halt=1.0)
        v = va.compute(self.root, EX)
        c = v["P3"]["per_config"]["m2d_dense-K4"]
        self.assertEqual(c["n_seeds_ge_095"], 2)
        self.assertEqual(c["n_cert_pass"], 0)
        self.assertEqual(c["status"], "not learnable")

    def test_m2d_last_has_no_certificate(self):
        full_root(self.root)
        ex = dict(EX, b_configs=EX["b_configs"] + [("m2d_last", 4)])
        write_b(self.root, "m2d_last", 4, om.build4(), halt=1.0)
        v = va.compute(self.root, ex)
        c = v["P3"]["per_config"]["m2d_last-K4"]
        self.assertTrue(c["n_cert_pass"].startswith("n/a"))
        self.assertTrue(c["learnable"])
        self.assertIn("accuracy condition only", c["learnable_basis"])

    def test_missing_artifacts_are_protocol_failures(self):
        full_root(self.root)
        os.remove(os.path.join(self.root, "b-m1d-k2-formal", f"s{SEEDS[0]}", "tables.json"))
        os.remove(os.path.join(self.root, "b-delay3-k5-formal", f"s{SEEDS[1]}", "test.json"))
        shutil.rmtree(os.path.join(self.root, "sat-m2d-n2-k4-last"))
        shutil.rmtree(os.path.join(self.root, "rA-d0-s500301-g4"))
        write_r(self.root, "P", 3, flip=False, complete=False)
        v = va.compute(self.root, EX)
        self.assertTrue(v["P1"]["status"].startswith("protocol failure (missing:"), v["P1"]["status"])
        self.assertIn("tables.json", v["P1"]["status"])
        self.assertTrue(v["P3"]["per_config"]["delay3-K5"]["status"].startswith("protocol failure"))
        self.assertTrue(v["P3"]["status"].startswith("protocol failure"))
        self.assertTrue(v["P4"]["status"].startswith("protocol failure"))
        self.assertTrue(v["C"]["status"].startswith("protocol failure"))
        self.assertTrue(v["R2"]["status"].startswith("protocol failure"))
        self.assertIn("rA-d0-s500301-g4", v["R2"]["status"])
        self.assertTrue(v["R1"]["status"].startswith("protocol failure"))  # incomplete P segment, not a non-flip
        self.assertIn("incomplete", v["R1"]["status"])

    def test_r_collapse_is_not_a_flip(self):
        full_root(self.root)
        write_r(self.root, "P", 1, flip=False, collapse=True)
        v = va.compute(self.root, EX)
        self.assertEqual(v["R1"]["flipped_runs"], [])
        self.assertTrue(v["R1"]["status"].startswith("no P run flipped"))

    def test_cli_writes_files(self):
        full_root(self.root)
        out = os.path.join(self.root, "_out")
        old = va.expected_default
        va.expected_default = lambda: EX
        try:
            va.main([self.root, "--out", out])
        finally:
            va.expected_default = old
        self.assertTrue(os.path.exists(os.path.join(out, "verdict.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "verdict.md")))

    def test_default_expectations_match_job_generator(self):
        ex = va.expected_default()
        jobs = {j["name"] for j in mj.build()["jobs"]}
        for t, K in ex["b_configs"]:
            self.assertIn(f"b-{t}-k{K}-formal", jobs)
        for t, s, d in ex["r_runs"]:
            self.assertIn(f"r{t}-d{d}-s{s}-g{ex['segments']}", jobs)
        for n in ex["sat"]:
            self.assertIn(n, jobs)
        self.assertEqual(len(ex["r_runs"]), 30)


if __name__ == "__main__":
    unittest.main()
