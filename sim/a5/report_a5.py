"""A5 report job: aggregate B (test.json/final.json per seed) and R-probe (summary.json/log.jsonl of final segments).

  python -m sim.a5.report_a5 --input in/report --out out/report

Reads in/report/<dep>/ for every dependency.  B jobs hold s<seed>/{final,test}.json; R final segments hold summary.json and
log.jsonl (full history, since each segment copies its predecessor's log).  Writes summary.json and summary.md.
Pure aggregation: no verdicts are computed here (the criteria are applied in the brief/report, not by this script),
except the R-probe criterion, which is the pre-registered train_r_probe.summarize_log.
"""
import argparse
import glob
import json
import os

from . import train_r_probe as rp


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    B, R = {}, {}
    for dep in sorted(glob.glob(os.path.join(a.input, "*"))):
        name = os.path.basename(dep)
        if name.startswith("b-"):
            runs = {}
            for t in sorted(glob.glob(os.path.join(dep, "s*", "test.json"))):
                d = os.path.dirname(t)
                test, fin = load(t), load(os.path.join(d, "final.json"))
                runs[os.path.basename(d)] = dict(dev_score=fin["dev_score"], best_step=fin["best_deploy_step"],
                                                 test={k: v["exact_halt"] for k, v in test["per_size"].items()},
                                                 test_fp={k: v["exact_fp"] for k, v in test["per_size"].items()},
                                                 family=test.get("family_a_2^(n-1)"))
            B[name] = runs
        elif name.startswith("r"):
            lp = os.path.join(dep, "log.jsonl")
            if os.path.exists(lp):
                with open(lp, encoding="utf-8") as f:
                    log = [json.loads(x) for x in f if x.strip()]
                R[name] = dict(config=load(os.path.join(dep, "config.json")), criterion=rp.summarize_log(log),
                               last=log[-1] if log else None)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(B=B, R=R), f, indent=1)
    lines = ["# A5 summary (auto-generated)", "", "## B runs: mean deployed (halted-and-correct) exact per test size", ""]
    for name, runs in B.items():
        if not runs:
            continue
        sizes = sorted(next(iter(runs.values()))["test"], key=int)
        mean = {n: sum(r["test"][n] for r in runs.values()) / len(runs) for n in sizes}
        lines.append(f"- {name}: seeds={len(runs)} " + " ".join(f"n{n}={mean[n]:.3f}" for n in sizes))
    lines += ["", "## R probe (final segment): criterion", ""]
    for name, r in R.items():
        c = r["criterion"]
        lines.append(f"- {name}: wd={r['config']['wd']:.3g} sustained={c['train_sustained_reached']} "
                     f"max32={c['max_hard_32_overall']:.3f} flipped={c['flipped']}")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
