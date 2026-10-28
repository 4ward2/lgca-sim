"""Stage A2 protocol runner (brief_A2.md §2-§4).

  python sim/sweep_a2.py tune  --task cycles            # R/F on a new task: K/alpha-beta grids x select seeds
  python sim/sweep_a2.py final --task cycles [--arms RFB]
  python sim/sweep_a2.py test  --task cycles [--arms RFB]   # --locked evaluation
  python sim/sweep_a2.py report
B has no tuning grid: `final` trains seeds 1..10 with sim/b_arm.py. For P/A/M, R and F are NOT rerun
(Stage A locked results are reused, brief_A2 §2).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep as S

ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs", "a2")
PY = sys.executable
G = 384


def common():
    return json.load(open(os.path.join(HERE, "common_a2.json"), encoding="utf-8-sig"))


def rf_cmd(task, arm, seed, cfg, out, threads):
    c = common()
    tc = c["per_task"].get(task, {})
    args = [PY, os.path.join(HERE, "train_a2.py"), "--task", task, "--arm", arm, "--G", str(G),
            "--seed", str(seed), "--out", out, "--threads", str(threads),
            "--lr", str(c["lr"]), "--steps", str(tc.get("steps", c["steps"])), "--resid", str(c["resid"]),
            "--mode", c["mode"], "--eval_every", str(c.get("eval_every", 250)), "--wd", str(c.get("wd", 0.0))]
    if tc.get("ckpt"):
        args.append("--ckpt")
    for key in ("wiring", "lconn", "xbias", "readout", "opt"):
        args += ["--" + key, str(c[key])]
    for flag in ("trim", "keep_best", "perch"):
        if c.get(flag):
            args.append("--" + flag)
    for k, v in cfg.items():
        args += [f"--{k}", str(v)]
    return args


def b_cmd(task, seed, out, threads):
    return [PY, os.path.join(HERE, "b_arm.py"), "train", "--task", task, "--seed", str(seed),
            "--out", out, "--threads", str(threads)]


def selected(task, arm):
    seeds = common()["select_seeds"][task]
    best, best_v, details = None, -1, []
    for cfg in S.tune_grid(arm):
        vs = [json.load(open(os.path.join(RUNS, "tune", S.name(task, arm, G, s, cfg), "final.json")))["val2"]["score"]
              for s in seeds]
        v = float(np.median(vs))
        details.append(dict(cfg=cfg, scores=vs, median=v))
        if v > best_v:
            best, best_v = cfg, v
    json.dump(details, open(os.path.join(RUNS, f"selected_{task}_{arm}.json"), "w"), indent=1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "final", "test", "report"])
    ap.add_argument("--task")
    ap.add_argument("--arms", default="RFB")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "report":
        import report_a2
        return report_a2.main()
    c = common()
    if not c.get("locked"):
        raise SystemExit("common_a2.json must set locked=true (brief_A2 lock) before running the protocol")
    t = a.task
    jobs = []
    if a.stage == "tune":
        assert t in c["rf_tasks"]
        for arm in "RF":
            for cfg in S.tune_grid(arm):
                for seed in c["select_seeds"][t]:
                    out = os.path.join(RUNS, "tune", S.name(t, arm, G, seed, cfg))
                    jobs.append((out, rf_cmd(t, arm, seed, cfg, out, a.threads), os.path.join(out, "final.json")))
    elif a.stage == "final":
        for arm in a.arms:
            if arm == "B" and t in c["b_tasks"]:
                for seed in c["final_seeds"]:
                    out = os.path.join(RUNS, "final", f"{t}_B_s{seed}")
                    jobs.append((out, b_cmd(t, seed, out, a.threads), os.path.join(out, "final.json")))
            elif arm in "RF" and t in c["rf_tasks"]:
                cfg = selected(t, arm)
                for seed in c["final_seeds"]:
                    out = os.path.join(RUNS, "final", S.name(t, arm, G, seed, cfg))
                    jobs.append((out, rf_cmd(t, arm, seed, cfg, out, a.threads), os.path.join(out, "final.json")))
    else:  # test: only runs that finished training
        fd = os.path.join(RUNS, "final")
        for d in sorted(os.listdir(fd)) if os.path.isdir(fd) else []:
            if not d.startswith(t + "_") or not os.path.exists(os.path.join(fd, d, "final.json")):
                continue
            arm = "B" if f"{t}_B_" in d else d[len(t) + 1]
            if arm not in a.arms:
                continue
            out = os.path.join(fd, d)
            if arm == "B":
                cmd = [PY, os.path.join(HERE, "b_arm.py"), "evaluate", out, "--locked", "--threads", str(a.threads)]
            else:
                cmd = [PY, os.path.join(HERE, "evaluate_a2.py"), out, "--locked", "--threads", str(a.threads)]
            jobs.append((out, cmd, os.path.join(out, "test.json")))
    S.run_pool(jobs, a.parallel)
    bad = [o for o, _, done in jobs if not os.path.exists(done)]
    if bad:
        print("FAILED/MISSING:", " ".join(os.path.basename(b) for b in bad), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
