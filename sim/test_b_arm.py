"""Tests for sim/b_arm.py (brief_A2 section 9.2). Validation-seed / training-stream data only; no test seeds,
no extrapolation sizes, the --locked path is never executed."""
import json
import math
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b_arm as ba  # noqa: E402
import tasks as tk  # noqa: E402
import tasks_a2 as t2  # noqa: E402
import evaluate as ev  # noqa: E402
import evaluate_a2 as ea  # noqa: E402
from logic import to_internal  # noqa: E402

torch.set_num_threads(2)
ROOT = os.path.dirname(HERE)
TMP = os.path.join(ROOT, "runs", "b_arm_test")


def close(a, b):
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return True
    return abs(float(a) - float(b)) < 1e-9


# ---------------------------------------------------------------- 1. hand-built oracles through the pipeline
def test_oracles():
    ref_bp = os.path.join(ROOT, "reviews", "a2", "fable", "oracle.json")
    ref_t = os.path.join(ROOT, "reviews", "a2", "fable", "runs", "T_oracle", "oracle_ternary.json")
    ref_bp = json.load(open(ref_bp)) if os.path.exists(ref_bp) else None
    ref_t = json.load(open(ref_t)) if os.path.exists(ref_t) else None
    with torch.no_grad():
        for task in ba.A1_TASKS:
            for infer in ("max", "sum"):
                sc, per = ba.score(ba.plant_bp_oracle(task, infer=infer), task)
                assert sc == 1.0, (task, infer, sc)
                if ref_bp is not None:  # bit for bit against Fable's earlier oracle.json
                    r = ref_bp[f"{task}/{infer}"]
                    assert close(r["score"], sc)
                    for n, v in r["per"].items():
                        p = per[n]
                        assert close(v["exact"], p["exact"]) and close(v["halt"], p["halt_rate"]), (task, infer, n)
                        assert close(v["t_halt"], p["mean_t_halt"]) and close(v["D"], p["mean_D"]), (task, infer, n)
            sc, per = ba.score(ba.plant_ternary_oracle(task), task)
            assert sc == 1.0, (task, "ternary", sc)
            assert all(v["halt_rate"] == 1.0 for v in per.values())
            if ref_t is not None:
                r = ref_t[task]
                assert close(r["score"], sc)
                for n, v in r["per"].items():
                    p = per[n]
                    assert close(v["exact"], p["exact"]) and close(v["t_halt"], p["mean_t_halt"]), (task, n)
                    assert close(v["t_fp"], p["t_fp"]) and close(v["D"], p["mean_D"]), (task, n)
            if task == "M":  # the ternary oracle is BFS: t_fp = D + 1 exactly
                for v in per.values():
                    assert close(v["t_fp"], v["mean_D"] + 1.0)
    print("oracles ok (BP max/sum and ternary = 1.0 on P/A/M" + (", bit for bit vs oracle.json" if ref_bp else "") + ")")


# ---------------------------------------------------------------- 2. four-state counting on a toy
def toy_maze():
    # 3x3: open left column with the source at the top; wall middle column; open right column = island
    open_ = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], np.float32)
    src = np.zeros((3, 3), np.float32); src[0, 0] = 1
    y = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], np.float32)
    return dict(x=np.stack([open_, src], 0)[None], y=y[None], mask=np.ones((1, 3, 3), bool), diff=np.array([2]))


def test_four_state():
    d = toy_maze()
    fs = ba.four_state_counts(ba.plant_ternary_oracle("M"), "M", d, 3)
    assert fs["states"] == list(ba.STATE_NAMES)
    assert fs["counts_by_label"] == [[0, 0], [3, 0], [0, 3], [3, 0]], fs  # walls {0}, reached {1}, island unknown
    assert fs["predicted_one_by_label"] == [[0, 0], [0, 0], [0, 3], [0, 0]], fs
    assert fs["n_cells"] == 9
    # the LLR-BP oracle has exact zeros on the island: LLR sign gives the same four states
    fs2 = ba.four_state_counts(ba.plant_bp_oracle("M", infer="max"), "M", d, 3)
    assert fs2["counts_by_label"] == fs["counts_by_label"], fs2
    # masked counting: only the mask is counted
    d2 = dict(d, mask=np.array([[[1, 1, 1], [0, 0, 0], [0, 0, 0]]], bool))
    fs3 = ba.four_state_counts(ba.plant_ternary_oracle("M"), "M", d2, 3)
    assert fs3["counts_by_label"] == [[0, 0], [1, 0], [0, 1], [1, 0]] and fs3["n_cells"] == 3, fs3
    print("four-state counting ok")


# ---------------------------------------------------------------- 3. adapter: run_hard == run_masked (all-ones mask)
def test_adapter_consistency():
    with torch.no_grad():
        for task, n, model in (("M", 8, ba.plant_ternary_oracle("M")), ("M", 8, ba.plant_bp_oracle("M", infer="max")),
                               ("A", 12, ba.plant_ternary_oracle("A"))):
            d = tk.fixed_set(task, n, 60, tk.VAL_SEED)
            d["mask"] = np.ones_like(d["y"], dtype=bool)
            a = ev.run_hard(model, "R", task, d, n, None, m=1, mode="hard")
            b = ea.run_masked(model, "R", tk.TASKS[task]["dims"], d, n, None, bs=200 if task != "M" else 100, m=1, mode="hard")
            for k in a:
                assert np.array_equal(a[k], b[k]), (task, type(model).__name__, k)
            # masked halting + scoring: labels and dynamics outside the mask must not matter
            rng = np.random.default_rng(1)
            mk = rng.random(d["y"].shape) < 0.4
            r2 = ea.run_masked(model, "R", tk.TASKS[task]["dims"], dict(d, mask=mk), n, None, m=1)
            r3 = ea.run_masked(model, "R", tk.TASKS[task]["dims"], dict(d, mask=mk, y=np.where(mk, d["y"], 1 - d["y"])), n, None, m=1)
            assert np.array_equal(r2["correct"], r3["correct"]) and np.allclose(r2["bitacc"], r3["bitacc"])
    # the ternary state is monotone: a step without change is the fixed point forever
    t = ba.plant_ternary_oracle("M")
    d = tk.fixed_set("M", 8, 20, tk.VAL_SEED)
    xv = to_internal(d["x"])
    with torch.no_grad():
        h = t.init_state(xv)
        prev = None
        for _ in range(64):
            hn = t.step(h, xv, "hard")
            c = hn[1:]
            if prev is not None:
                assert (c <= prev).all()  # possibilities only disappear
            prev = c
            h = hn
        assert torch.equal(t.step(h, xv, "hard"), h)
    print("adapter consistency ok (run_hard == run_masked with all-ones mask; masked scoring; monotone state)")


# ---------------------------------------------------------------- 4. smoke train + hook evaluation (in-distribution only)
def test_smoke_train():
    shutil.rmtree(TMP, ignore_errors=True)
    for task, n_val in (("A", 200), ("cycles_reach", 40)):
        out = os.path.join(TMP, task)
        res = ba.main(["train", "--task", task, "--seed", "0", "--out", out, "--threads", "2",
                       "--steps", "100", "--eval_every", "50", "--n_val", str(n_val)])
        cfg = json.load(open(os.path.join(out, "config.json")))
        assert cfg["deploy"] == "ternary" and cfg["thr"] == 2.0 and cfg["lr"] == 0.03 and cfg["beta"] == 2.0
        assert cfg["batch"] == (128 if task == "A" else 32)
        assert set(cfg["sha256"]) == {"tasks.py", "tasks_a2.py", "evaluate.py", "evaluate_a2.py", "b_arm.py"}
        assert "git_commit" in cfg
        for name in ("best_deploy.pt", "best_soft.pt", "last.pt", "final.json", "log.json"):
            assert os.path.exists(os.path.join(out, name)), name
        fin = json.load(open(os.path.join(out, "final.json")))
        assert set(fin["checkpoints"]) == {"best_deploy", "best_soft", "last"}
        for ck in fin["checkpoints"].values():
            assert "val" in ck and "val2" in ck and 0.0 <= ck["val"]["score"] <= 1.0
        assert fin["val"] == fin["checkpoints"]["best_deploy"]["val"]
        assert fin["four_state_val"]["n_cells"] > 0 and fin["M4"]["lut6_lower_bound"] > 0
        if task == "cycles_reach":
            assert set(fin["small_graph_diag"]) == {"5", "6"}
            for v in fin["small_graph_diag"].values():
                assert v["cells"] == v["graphs"] * int(list(fin["small_graph_diag"].keys())[0]) ** 2 or v["cells"] > 0
        # the evaluate path, exercised on validation-seed, in-distribution data only (test hook)
        hook = ba.main(["evaluate", out, "--hook", "--threads", "2"])
        hj = json.load(open(os.path.join(out, "test_hook.json")))
        assert hj["hook"] is True and hj["test_seed"] == 777
        for k in ("M1", "M2", "bar", "M1_v2", "M2_ws", "M1_w8", "M2_wT4", "bar_wTmax", "outcome_frac_extra", "per_size",
                  "M4", "best_soft_M2", "last_M2", "best_soft_softmode_M2", "four_state", "t_fp", "rho_halt", "checkpoint_steps"):
            assert k in hj, k
        assert hj["M1"] == hj["M2"] == hj["bar"]  # hook: only the in-distribution size
        ind = "16" if task == "A" else "12"
        assert list(hj["per_size"]) == [ind]
        assert ind in hj["four_state"]
        assert 12345 not in (hj["test_seed"],) and 24680 not in (hj["test_seed"],)
        print(f"smoke {task}: best_deploy step {fin['best_deploy_step']} val {fin['val']['score']:.3f} "
              f"soft {fin['checkpoints']['best_soft']['val']['score']:.3f} hook M1 {hj['M1']:.3f}")
    # diag subcommand on the Q+ run
    dg = ba.main(["diag", os.path.join(TMP, "cycles_reach"), "--count", "20", "--threads", "2"])
    assert set(dg) == {"5", "6"} and all("by_tail" in v for v in dg.values())
    print("smoke train + hook evaluation + diag ok")


if __name__ == "__main__":
    test_four_state()
    test_adapter_consistency()
    test_oracles()
    test_smoke_train()
    print("b_arm tests ok")
