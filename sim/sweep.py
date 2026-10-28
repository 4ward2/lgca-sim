"""Run the Stage A protocol (brief_A.md section 3) locally.

  python sim/sweep.py tune  --task P --G 384      # 4 configs per arm, seed 0, in-distribution val only
  python sim/sweep.py final --task P --G 384      # best config per arm, seeds 1,2,3 (+ arm C at G_small)
  python sim/sweep.py test  --task P --G 384      # --locked evaluation of the final runs
  python sim/sweep.py report                      # decision table from test.json files

Common settings (lr, steps, resid, mode, ...) come from sim/common.json, which is written to
NOTES.md before the locked run.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
RUNS = os.path.join(ROOT, "runs")
G_SMALL, G_LARGE = 384, 1536
# brief_A.md section 9 (amendment v2): 10 seeds for P/A, 5 for M
FINAL_SEEDS = {"P": tuple(range(1, 11)), "A": tuple(range(1, 11)), "M": tuple(range(1, 6))}


def common():
    return json.load(open(os.path.join(HERE, "common.json"), encoding="utf-8-sig"))


def tune_grid(arm):
    if arm == "F":
        return [dict(K=k) for k in (1, 2, 4, 8)]
    return [dict(alpha=al, beta=be) for al, be in itertools.product((0.0, 0.5), (2.0, 4.0))]


def name(task, arm, G, seed, cfg):
    tag = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    return f"{task}_{arm}_G{G}_{tag}_s{seed}" if tag else f"{task}_{arm}_G{G}_s{seed}"


def train_cmd(task, arm, G, seed, cfg, out, threads):
    c = common()
    task_c = c.get("per_task", {}).get(task, {})
    args = [PY, os.path.join(HERE, "train.py"), "--task", task, "--arm", arm, "--G", str(G),
            "--seed", str(seed), "--out", out, "--threads", str(threads),
            "--lr", str(c["lr"]), "--steps", str(task_c.get("steps", c["steps"])),
            "--resid", str(c["resid"]), "--mode", c["mode"],
            "--eval_every", str(c.get("eval_every", 1000)), "--wd", str(c.get("wd", 0.0))]
    if c.get("ckpt") or task_c.get("ckpt"):
        args.append("--ckpt")
    for key in ("wiring", "lconn", "xbias", "readout", "trim", "opt"):
        if key in c:
            if isinstance(c[key], bool):
                if c[key]: args.append("--" + key)
            else:
                args += ["--" + key, str(c[key])]
    if c.get("keep_best"): args.append("--keep_best")
    if c.get("perch"): args.append("--perch")
    for k, v in cfg.items():
        args += [f"--{k}", str(v)]
    return args


def run_pool(jobs, parallel):
    """jobs: list of (out_dir, argv). Skips jobs whose final.json/test.json already exists."""
    pending = list(jobs)
    live = []
    while pending or live:
        while pending and len(live) < parallel:
            out, argv, done_file = pending.pop(0)
            if os.path.exists(done_file):
                continue
            os.makedirs(out, exist_ok=True)
            logf = open(os.path.join(out, "stdout.log"), "a")
            live.append((subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT), out, logf))
            print(time.strftime("%H:%M:%S"), "start", os.path.basename(out), flush=True)
        time.sleep(5)
        for p, out, logf in list(live):
            if p.poll() is not None:
                logf.close()
                live.remove((p, out, logf))
                print(time.strftime("%H:%M:%S"), "exit", p.returncode, os.path.basename(out), flush=True)


def selected(task, G, arm):
    """Best tuning config by median val2 score (ties -> first in grid)."""
    best, best_v = None, -1
    details = []
    seeds = common().get("select_seeds", {}).get(task, [0])
    for cfg in tune_grid(arm):
        vs=[]
        for seed in seeds:
            f = os.path.join(RUNS, "tune", name(task, arm, G, seed, cfg), "final.json")
            vs.append(json.load(open(f))["val2"]["score"])
        v = float(np.median(vs)); details.append(dict(cfg=cfg, scores=vs, median=v))
        if v > best_v:
            best, best_v = cfg, v
    json.dump(details, open(os.path.join(RUNS, f"selected_{task}_G{G}_{arm}.json"), "w"), indent=1)
    return best, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "final", "test", "report"])
    ap.add_argument("--task", default="P")
    ap.add_argument("--G", type=int, default=G_SMALL)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arms", default="RF")
    a = ap.parse_args()
    t, G = a.task, a.G
    if a.stage in ("tune", "final", "test") and not common().get("locked", False):
        raise SystemExit("common.json must set locked=true before this stage")
    if a.stage == "tune":
        jobs = []
        for arm in a.arms:
            if arm == "C":
                continue
            for cfg in tune_grid(arm):
                for seed in common().get("select_seeds", {}).get(t, [0]):
                    out = os.path.join(RUNS, "tune", name(t, arm, G, seed, cfg))
                    jobs.append((out, train_cmd(t, arm, G, seed, cfg, out, a.threads), os.path.join(out, "final.json")))
        run_pool(jobs, a.parallel)
    elif a.stage == "final":
        jobs = []
        sel = {}
        for arm in a.arms:
            if arm == "C":
                if G != G_SMALL:
                    continue
                cfg = {}
            else:
                cfg, v = selected(t, G, arm)
                sel[arm] = dict(cfg=cfg, val=v)
            for seed in FINAL_SEEDS[t]:
                out = os.path.join(RUNS, "final", name(t, arm, G, seed, cfg))
                jobs.append((out, train_cmd(t, arm, G, seed, cfg, out, a.threads),
                             os.path.join(out, "final.json")))
        print("selected:", json.dumps(sel))
        json.dump(sel, open(os.path.join(RUNS, f"selected_{t}_G{G}.json"), "w"), indent=1)
        run_pool(jobs, a.parallel)
    elif a.stage == "test":
        jobs = []
        for d in sorted(os.listdir(os.path.join(RUNS, "final"))):
            if not d.startswith(f"{t}_") or f"_G{G}_" not in d + "_":
                continue
            out = os.path.join(RUNS, "final", d)
            jobs.append((out, [PY, os.path.join(HERE, "evaluate.py"), out, "--locked",
                               "--threads", str(a.threads)], os.path.join(out, "test.json")))
        run_pool(jobs, a.parallel)
    else:
        import report
        report.main()


if __name__ == "__main__":
    main()
