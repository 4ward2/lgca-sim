"""A5 verdict script: applies the pre-registered decision rules of brief_A5.md (P1, P2, P3, P4, C, R1, R2) to the
downloaded job artifacts.   python -m sim.a5.verdict_a5 ROOT --out DIR

ROOT holds one sub-directory per job (as downloaded by tools/gh_dag.py into in/verdict/):
  b-<task>-k<K>-formal/s<seed>/{test.json,tables.json,final.json}
  r<A|P>-d<decay>-s<seed>-g4/{summary.json,log.jsonl}            (final R segment; the log holds the full history)
  sat-*/*.result.json                                            (k2_sat CLI outcome: status SAT|UNSAT|TIMEOUT, ...)
Writes DIR/verdict.json and DIR/verdict.md.  A missing / incomplete artifact makes the status of every rule that depends on it
"protocol failure (missing: ...)"; nothing is silently counted as a failure or a pass.

Rules (all quantities recomputed here from the stored artifacts; only test.json's exact_halt numbers are taken from the run):
  P1  every formal m1d table: EXHAUSTIVE fixed-point exact match on the a = 2^(n-1) family (all 2^n b) at n in {8,12,16} <=
      2^(K+2-n), and m1d_cert.smallest_counterexample gives a simulator-confirmed counterexample with n <= K+3.
  P2  delay3 K=1 and delay4 K=2 formal tables: delay.pigeonhole_counterexample exists and is simulator-confirmed.
  P3  per config (m2d_dense, m2d_last, delay3, delay4): seeds with exact_halt >= 0.95 at every test size; certificate PASS count
      (wave_cert for m2d_dense, delay_cert for delay, "n/a (dense-only certificate)" for m2d_last); fraction of cells with a
      non-singleton fixed-point set (ac.py on the stored tables, 200 inputs per test size from seed 500001).
      learnable <=> >= 8/10 seeds reach 0.95 at all test sizes AND >= 8/10 certificates PASS (m2d_last: first condition only,
      flagged).
  P4  m2d K=2: SAT statuses (n in {2,3}, last/all, plus the n=2 K=4 control) and the count of K=2 seeds reaching >= 0.95 at
      all test sizes (m2d_dense and m2d_last).  Prediction: none.
  C   (learnability form) supported <=> P1 holds for all tables AND m2d_dense K=4 is learnable.
  R1  some positive-control (P) run flipped;  R2  some A run flipped  (flip rule: train_r_probe.summarize_log).
"""
import argparse
import glob
import json
import os

import numpy as np

from . import (ac, delay, delay_cert, m1d_cert, make_jobs_a5 as mj, rounding_k as rk, train_b as tb,
               train_r_probe as rp, wave_cert)

P1_SIZES = (8, 12, 16)
NS_SEED, NS_COUNT = 500001, 200
THRESH, NEED_SEEDS = 0.95, 8


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def expected_default():
    """Expected artifacts, from the job generator (single source of truth)."""
    sat = [n for n, _ in mj.SAT_JOBS]
    return dict(b_configs=[(t, K) for t, ks in mj.B_CONFIGS for K in ks], seeds=list(mj.FORMAL_SEEDS),
                r_runs=[(t, s, d) for d in mj.DECAYS for t in ("A", "P") for s in (mj.A_SEEDS if t == "A" else mj.P_SEEDS)],
                segments=mj.SEGMENTS, sat=sat)


def _fail(missing):
    shown = ", ".join(missing[:8]) + (f", ... (+{len(missing) - 8} more)" if len(missing) > 8 else "")
    return f"protocol failure (missing: {shown})"


def b_dir(root, task, K):
    return os.path.join(root, f"b-{task}-k{K}-formal")


def load_config(root, task, K, seeds, need=("test.json", "tables.json", "final.json")):
    """-> (runs {seed: dict(test, tables)}, missing [paths])"""
    runs, missing = {}, []
    for s in seeds:
        d = os.path.join(b_dir(root, task, K), f"s{s}")
        absent = [f"b-{task}-k{K}-formal/s{s}/{f}" for f in need if not os.path.exists(os.path.join(d, f))]
        if absent:
            missing += absent
            continue
        runs[s] = dict(test=_load(os.path.join(d, "test.json")) if "test.json" in need else None,
                       tables=rk.from_json(_load(os.path.join(d, "tables.json"))) if "tables.json" in need else None)
    return runs, missing


def reaches(test, thr=THRESH):
    return all(v["exact_halt"] >= thr for v in test["per_size"].values())


# ------------------------------------------------------------------------------------------------ P1 / P2
def rule_p1(root, ex):
    per, missing = {}, []
    for task, K in ex["b_configs"]:
        if task != "m1d":
            continue
        runs, miss = load_config(root, task, K, ex["seeds"], need=("tables.json",))
        missing += miss
        for s, r in runs.items():
            T = r["tables"]
            fam = {}
            for n in P1_SIZES:
                m = m1d_cert.family_exact_match(T, n)
                bound = m1d_cert.family_bound(K, n)
                m.update(bound=bound, ok=m["correct"] <= 2 ** (K + 2) and m["acc"] <= min(1.0, bound) + 1e-12)
                fam[str(n)] = m
            cx = m1d_cert.smallest_counterexample(T)
            cert = dict(n=cx["n"], method=cx["method"], simulator_agrees=cx["simulator_agrees"], b=cx["b"],
                        ok=bool(cx["n"] <= K + 3 and cx["simulator_agrees"]))
            per[f"K{K}/s{s}"] = dict(family=fam, counterexample=cert,
                                     ok=all(v["ok"] for v in fam.values()) and cert["ok"])
    if missing:
        return dict(status=_fail(missing), per_table=per)
    bad = [k for k, v in per.items() if not v["ok"]]
    return dict(status="holds" if not bad else f"VIOLATED (bug, stop and audit theory/scorer): {bad}", per_table=per)


def rule_p2(root, ex):
    per, missing = {}, []
    for task, K, d in (("delay3", 1, 3), ("delay4", 2, 4)):
        if (task, K) not in ex["b_configs"]:
            continue
        runs, miss = load_config(root, task, K, ex["seeds"], need=("tables.json",))
        missing += miss
        for s, r in runs.items():
            cx = delay.pigeonhole_counterexample(r["tables"], d)
            ok = bool(cx is not None and cx["confirmed_by_simulator"])
            per[f"{task}-K{K}/s{s}"] = dict(ok=ok, found=cx is not None,
                                           confirmed=None if cx is None else cx["confirmed_by_simulator"])
    if missing:
        return dict(status=_fail(missing), per_table=per)
    bad = [k for k, v in per.items() if not v["ok"]]
    return dict(status="holds" if not bad else f"VIOLATED (bug): {bad}", per_table=per)


# ------------------------------------------------------------------------------------------------ P3
def nonsingleton_fraction(task, tabs, seed=NS_SEED, count=NS_COUNT):
    """Fraction of cells whose AC fixed-point set is not a singleton (empty sets count as non-singleton), per test size and
    overall, on the fixed deterministic sample tb.fixed_set(task, n, count, seed)."""
    per = {}
    for n in tb.TASKS[task]["test"]:
        d = tb.fixed_set(task, n, count, seed)
        xi = ac.codes_from_x(d["x"])
        c, _ = ac.fixed_point(tabs["u"], tabs["r"], xi, tmax=4 * tb.t_max(d["x"].shape) + 4)
        per[str(n)] = float((c.sum(-1) != 1).mean())
    return dict(per_size=per, mean=float(np.mean(list(per.values()))))


def certificate(task, tabs):
    if task == "m2d_dense":
        return wave_cert.certify(tabs, "all", search=False)["status"]
    if task in ("delay3", "delay4"):
        return delay_cert.certify(tabs, tb.TASKS[task]["d"])["status"]
    return "n/a (dense-only certificate)"


def p3_config(root, ex, task, K):
    runs, missing = load_config(root, task, K, ex["seeds"])
    if missing:
        return dict(status=_fail(missing))
    n_seeds = len(runs)
    acc = {s: reaches(r["test"]) for s, r in runs.items()}
    certs = {s: certificate(task, r["tables"]) for s, r in runs.items()}
    ns = {s: nonsingleton_fraction(task, r["tables"]) for s, r in runs.items()}
    n_acc = sum(acc.values())
    out = dict(seeds=n_seeds, n_seeds_ge_095=n_acc, per_seed_ge_095={str(s): v for s, v in acc.items()},
               certificates={str(s): v for s, v in certs.items()},
               nonsingleton_frac_mean=float(np.mean([v["mean"] for v in ns.values()])),
               nonsingleton_frac_per_seed={str(s): v["mean"] for s, v in ns.items()})
    need = NEED_SEEDS if n_seeds == 10 else -(-NEED_SEEDS * n_seeds // 10)  # 8/10; scaled up only for reduced test setups
    if task == "m2d_last":
        out.update(n_cert_pass="n/a (dense-only certificate)", learnable=bool(n_acc >= need),
                   learnable_basis="accuracy condition only (no certificate for m2d_last; flagged)")
    else:
        n_pass = sum(v == "PASS" for v in certs.values())
        out.update(n_cert_pass=n_pass, learnable=bool(n_acc >= need and n_pass >= need),
                   learnable_basis="accuracy AND certificate")
    out["status"] = "learnable" if out["learnable"] else "not learnable"
    return out


P3_CONFIGS = (("m2d_dense", (4, 16)), ("m2d_last", (4, 16)), ("delay3", (5, 8)), ("delay4", (6, 16)))


def rule_p3(root, ex):
    per = {}
    for task, ks in P3_CONFIGS:
        for K in ks:
            if (task, K) in ex["b_configs"]:
                per[f"{task}-K{K}"] = p3_config(root, ex, task, K)
    miss = [k for k, v in per.items() if v["status"].startswith("protocol failure")]
    return dict(status="reported" if not miss else _fail(miss), per_config=per)


# ------------------------------------------------------------------------------------------------ SAT / P4
def load_sat(root, names):
    out, missing = {}, []
    for n in names:
        fs = sorted(glob.glob(os.path.join(root, n, "*.result.json")))
        if not fs:
            missing.append(f"{n}/*.result.json")
            continue
        r = _load(fs[0])
        out[n] = dict(status=r["status"], n=r.get("n"), K=r.get("K"), d=r.get("d"), targets=r.get("targets"),
                      vars=r.get("vars"), clauses=r.get("clauses"), seconds=r.get("seconds"))
    return out, missing


def rule_p4(root, ex):
    names = [n for n in ex["sat"] if n.startswith("sat-m2d")]
    sat, miss = load_sat(root, names)
    cfgs, bmiss = {}, []
    for task in ("m2d_dense", "m2d_last"):
        if (task, 2) in ex["b_configs"]:
            runs, m = load_config(root, task, 2, ex["seeds"], need=("test.json",))
            bmiss += m
            cfgs[f"{task}-K2"] = sum(reaches(r["test"]) for r in runs.values())
    if miss or bmiss:
        return dict(status=_fail(miss + bmiss), sat=sat, k2_seeds_ge_095=cfgs)
    note = []
    ctrl = sat.get("sat-m2d-n2-k4-last")
    if ctrl is not None and ctrl["status"] != "SAT":
        note.append("control sat-m2d-n2-k4-last is not SAT: theory/encoding bug")
    holds = all(v == 0 for v in cfgs.values())
    return dict(status="prediction holds (no K=2 seed reaches 0.95 at all test sizes)" if holds
                else "prediction refuted (some K=2 seed reaches 0.95 at all test sizes)",
                sat=sat, k2_seeds_ge_095=cfgs, notes=note)


def rule_sat_delay(root, ex):
    names = [n for n in ex["sat"] if n.startswith("sat-delay")]
    sat, miss = load_sat(root, names)
    return dict(status=_fail(miss) if miss else "reported", sat=sat)


def rule_c(p1, p3):
    p3c = p3["per_config"].get("m2d_dense-K4", dict(status="protocol failure (missing: m2d_dense-K4)"))
    miss = [r["status"] for r in (p1, p3c) if r["status"].startswith("protocol failure")]
    if miss:
        return dict(status="; ".join(miss))
    p1_ok = p1["status"] == "holds"
    lr = p3c["learnable"]
    return dict(status="supported" if (p1_ok and lr) else "not supported", P1_holds=p1_ok, m2d_dense_K4_learnable=lr,
                note="representability part (theorem 2 + wave_cert PASS on the K=4 oracle) is established before lock and "
                     "is not an A5 result")


# ------------------------------------------------------------------------------------------------ R1 / R2
def load_r(root, runs, segments):
    out, missing = {}, []
    for task, s, d in runs:
        name = f"r{task}-d{d}-s{s}-g{segments}"
        lp, sp = os.path.join(root, name, "log.jsonl"), os.path.join(root, name, "summary.json")
        if not (os.path.exists(lp) and os.path.exists(sp)):
            missing.append(name)
            continue
        summ = _load(sp)
        if not summ.get("complete", False):
            missing.append(name + "(incomplete)")
            continue
        with open(lp, encoding="utf-8") as f:
            log = [json.loads(x) for x in f if x.strip()]
        out[name] = dict(task=task, seed=s, decay=d, wd=summ.get("wd"), criterion=rp.summarize_log(log))
    return out, missing


def rule_r(root, ex):
    runs, missing = load_r(root, ex["r_runs"], ex["segments"])
    res = {}
    for key, task in (("R1", "P"), ("R2", "A")):
        mine = {k: v for k, v in runs.items() if v["task"] == task}
        miss = [m for m in missing if m.startswith(f"r{task}-")]
        res[key] = dict(status=_fail(miss) if miss else None,
                        flipped_runs=[k for k, v in mine.items() if v["criterion"]["flipped"]],
                        runs={k: dict(flipped=v["criterion"]["flipped"], sustained=v["criterion"]["train_sustained_reached"],
                                      max_hard_32=v["criterion"]["max_hard_32_overall"]) for k, v in mine.items()})
    if res["R1"]["status"] is None:
        res["R1"]["status"] = "detection power shown" if res["R1"]["flipped_runs"] else \
            "no P run flipped: probe has no shown detection power (A result uninformative)"
    if res["R2"]["status"] is None:
        informative = bool(res["R1"]["flipped_runs"])
        res["R2"]["status"] = ("observed" if res["R2"]["flipped_runs"] else "not observed") + \
                              ("" if informative else " (uninformative: R1 did not show detection power)")
    return res


# ------------------------------------------------------------------------------------------------ driver
def compute(root, ex=None):
    ex = ex or expected_default()
    p1, p2, p3 = rule_p1(root, ex), rule_p2(root, ex), rule_p3(root, ex)
    r = rule_r(root, ex)
    return dict(root=os.path.abspath(root), P1=p1, P2=p2, P3=p3, P4=rule_p4(root, ex), SAT_delay=rule_sat_delay(root, ex),
                C=rule_c(p1, p3), R1=r["R1"], R2=r["R2"])


def to_markdown(v):
    L = ["# A5 verdict (auto-generated by sim/a5/verdict_a5.py)", ""]
    for k in ("P1", "P2", "P3", "P4", "SAT_delay", "C", "R1", "R2"):
        L.append(f"- **{k}**: {v[k]['status']}")
    L += ["", "## P3 per config", ""]
    for name, c in v["P3"]["per_config"].items():
        if "learnable" not in c:
            L.append(f"- {name}: {c['status']}")
            continue
        L.append(f"- {name}: seeds>=0.95 = {c['n_seeds_ge_095']}/{c['seeds']}, certificate PASS = {c['n_cert_pass']}, "
                 f"non-singleton cell fraction = {c['nonsingleton_frac_mean']:.4f}, **{c['status']}** ({c['learnable_basis']})")
    L += ["", "## SAT", ""]
    for key in ("P4", "SAT_delay"):
        for n, s in v[key].get("sat", {}).items():
            L.append(f"- {n}: {s['status']} (vars {s['vars']}, clauses {s['clauses']}, {s['seconds']} s)")
    if "k2_seeds_ge_095" in v["P4"]:
        L += ["", "P4 K=2 seeds reaching 0.95 at all test sizes: " + json.dumps(v["P4"]["k2_seeds_ge_095"])]
    L += ["", "## R", ""]
    for k in ("R1", "R2"):
        L.append(f"- {k} flipped runs: {v[k]['flipped_runs']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    v = compute(a.root)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1)
    with open(os.path.join(a.out, "verdict.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(v))
    print(to_markdown(v))


if __name__ == "__main__":
    main()
