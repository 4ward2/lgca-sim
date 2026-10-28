"""Stage A2 decision table (brief_A2.md §6), implemented literally.

Inputs:
  R, F on P/A/M  : Stage A locked results  runs/final/<task>_<arm>_G<G>_*/test.json   (reused, §2)
  R, F on new    : runs/a2/final/<task>_<arm>_G<G>_*/test.json                         (train_a2/evaluate_a2)
  B everywhere   : runs/a2/final/<task>_B_s<seed>/test.json                            (b_arm.py)
Primary comparison budget: G = 384 (brief_A2 §6, 'COMPARE_G'). G = 1536 is a secondary comparison and
exists only for P/A/M (Stage A); the new tasks have R/F at G = 384 only.
Run validity (F1/F4 of the astra final review):
  * a run whose test.json/final.json is missing or unreadable, or whose M1/M2/bar is missing or
    non-finite, is a FAILED run: all its metrics count as 0 and it is listed with the reason;
  * B runs must carry config.json with the directory's seed and the frozen budget (2000 steps, eval 250);
  * two directories for the same (task, arm, G, seed) make that (task, arm, G) a protocol failure;
  * a secondary-criterion field missing in otherwise valid runs makes that criterion 'n/a'.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL_A = os.path.join(ROOT, "runs", "final")
FINAL_A2 = os.path.join(ROOT, "runs", "a2", "final")
MAIN = ("P", "A", "M")
PRIOR = ("cycles_reach", "cycles")      # Q+, Q
NEW = ("inpaint", "life", "midframe")
SEEDS = tuple(range(1, 11))             # brief_A2 §4: 10 final seeds for every (task, arm)
COMPARE_G = 384
NAME = re.compile(r"^(?P<task>[A-Za-z_]+?)_(?P<arm>[RFCB])(?:_G(?P<G>\d+))?_.*?s(?P<seed>\d+)$")
CRITERIA = {"primary": "", "v2": "_v2", "ws": "_ws", "wT4": "_wT4", "wTmax": "_wTmax"}
REQUIRED = ("M1", "M2", "bar")
EPS = 1e-9  # brief_A2 §6: equality counts as a win; fixed float tolerance
B_BUDGET = dict(steps=2000, eval_every=250)
NA_B = {"midframe": "N/A (B structurally unrepresentable; not run, brief_A2 §2)"}


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))


def _check_run(final_dir, d, seed, arm):
    """Returns (record or None, reason). None = failed run."""
    f, fj = os.path.join(final_dir, d, "test.json"), os.path.join(final_dir, d, "final.json")
    if not (os.path.exists(f) and os.path.exists(fj)):
        return None, "missing test.json/final.json"
    try:
        r = json.load(open(f))
    except Exception as e:
        return None, f"unreadable test.json ({type(e).__name__})"
    if not isinstance(r, dict):
        return None, "test.json is not an object"
    for k in REQUIRED:
        if not _finite(r.get(k)):
            return None, f"{k} missing or non-finite ({r.get(k)!r})"
    if arm == "B":
        try:
            cfg = json.load(open(os.path.join(final_dir, d, "config.json")))
        except Exception:
            return None, "missing/unreadable config.json"
        if int(cfg.get("seed", -1)) != seed:
            return None, f"config seed {cfg.get('seed')} != directory seed {seed}"
        if any(int(cfg.get(k, -1)) != v for k, v in B_BUDGET.items()):
            return None, f"B budget mismatch (steps={cfg.get('steps')}, eval_every={cfg.get('eval_every')})"
    return r, ""


def _load(final_dir, tasks, problems):
    rows = defaultdict(dict)
    seen = defaultdict(list)
    if not os.path.isdir(final_dir):
        return rows
    for d in sorted(os.listdir(final_dir)):
        m = NAME.match(d)
        if not m or m["task"] not in tasks:
            continue
        key = (m["task"], m["arm"], int(m["G"]) if m["G"] else None)
        seed = int(m["seed"])
        seen[key + (seed,)].append(d)
        r, why = _check_run(final_dir, d, seed, m["arm"])
        if r is None:
            problems.setdefault("failed_runs", []).append(f"{d}: {why}")
        rows[key][seed] = r
    for k, ds in seen.items():
        if len(ds) > 1:
            problems.setdefault("duplicates", []).append(f"{k}: {ds}")
            problems.setdefault("dup_keys", set()).add(k[:3])
    return rows


def collect(final_a=FINAL_A, final_a2=FINAL_A2, problems=None):
    problems = {} if problems is None else problems
    rows = defaultdict(dict)
    for k, v in _load(final_a, MAIN, problems).items():          # Stage A locked R/F (C ignored)
        if k[1] in "RF":
            rows[k].update(v)
    for k, v in _load(final_a2, MAIN + PRIOR + NEW, problems).items():
        if k[1] == "B" or k[0] not in MAIN:                       # B everywhere; R/F only for new tasks
            rows[k].update(v)
    return rows


def _val(r, key):
    """Failed run -> 0. Absent/non-finite secondary field in a valid run -> None (criterion n/a)."""
    if r is None:
        return 0.0
    v = r.get(key)
    return float(v) if _finite(v) else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(rows, seeds=SEEDS, suffix="", dup_keys=()):
    out = {}
    for key, per in rows.items():
        complete = all(s in per for s in seeds) and key not in dup_keys
        rs = [per.get(s) for s in seeds]
        vals = {k: [_val(r, k + suffix) for r in rs] for k in REQUIRED}
        available = all(v is not None for vs in vals.values() for v in vs)
        med = {k: (float(np.median(vs)) if available else None) for k, vs in vals.items()}
        ks = sum((_val(r, "M1") or 0.0) >= 0.9 - EPS for r in rs)
        out[key] = dict(med_M1=med["M1"], med_M2=med["M2"], med_bar=med["bar"], complete=complete,
                        failed=sum(r is None for r in rs), available=available,
                        success=ks / len(seeds), success_ci=wilson(ks, len(seeds)))
    return out


def _get(summ, task, arm, G):
    return summ.get((task, arm, G if arm != "B" else None))


def win(summ, task, arm, G=COMPARE_G):
    """win(t, X) := med_M1 >= 0.9 and med_M2(X) - med_M2(F) >= 0.10 (with EPS).
    None = protocol failure (missing/incomplete/duplicate); 'n/a' = criterion not available."""
    if arm == "B" and task in NA_B:
        return NA_B[task]
    X, Fs = _get(summ, task, arm, G), _get(summ, task, "F", G)
    if X is None or Fs is None or not X["complete"] or not Fs["complete"]:
        return None
    if not (X["available"] and Fs["available"]):
        return "n/a"
    return bool(X["med_M1"] >= 0.9 - EPS and (X["med_M2"] - Fs["med_M2"]) >= 0.10 - EPS)


def conclusions(summ, G=COMPARE_G):
    w = {t: win(summ, t, "B", G) for t in MAIN + PRIOR + NEW}
    if any(w[t] == "n/a" for t in MAIN):
        main = "n/a (criterion not available)"
    elif any(w[t] is None for t in MAIN):
        main = "protocol failure (incomplete main-task results)"
    else:
        votes = sum(w[t] for t in MAIN)
        main = "POSITIVE" if votes >= 2 and (w["A"] or w["M"]) else "NEGATIVE"
    prior = None
    if main == "POSITIVE":
        q, qp = w["cycles"], w["cycles_reach"]
        if q is None or qp is None or "n/a" in (q, qp):
            prior = "protocol failure (Q/Q+ incomplete)"
        elif qp and q:
            prior = "rule-learning positive"
        elif qp:
            prior = "prior-matched positive"
        else:
            prior = "Q control invalid (B fails the positive control Q+)"
    usable = any((_get(summ, t, "B", G)["med_bar"] or 0.0) >= 0.9 - EPS for t in MAIN + PRIOR + NEW
                 if _get(summ, t, "B", G) and _get(summ, t, "B", G)["complete"] and _get(summ, t, "B", G)["available"])
    lines = {t: dict(B=w[t], R=win(summ, t, "R", G)) for t in NEW}

    def b_minus_r(t):
        b, r = _get(summ, t, "B", G), _get(summ, t, "R", G)
        if not (b and r) or b["med_M2"] is None or r["med_M2"] is None:
            return None
        return b["med_M2"] - r["med_M2"]
    return dict(win_B=w, main=main, prior=prior, usable=usable, new_lines=lines,
                B_minus_R={t: b_minus_r(t) for t in MAIN + PRIOR + NEW})


def main(final_a=FINAL_A, final_a2=FINAL_A2, write=True, quiet=False):
    problems = {}
    rows = collect(final_a, final_a2, problems)
    dup = problems.get("dup_keys", set())
    res = {}
    for name, suf in CRITERIA.items():
        summ = summarize(rows, suffix=suf, dup_keys=dup)
        res[name] = {f"G{G}": conclusions(summ, G) for G in (384, 1536)}
    prim = res["primary"][f"G{COMPARE_G}"]
    others = {n: r[f"G{COMPARE_G}"]["main"] for n, r in res.items()
              if not r[f"G{COMPARE_G}"]["main"].startswith("n/a")}
    headline = prim["main"] if len(set(others.values())) <= 1 else \
        "criterion-dependent: " + ", ".join(f"{k}={v}" for k, v in others.items())
    summ = summarize(rows, dup_keys=dup)
    if not quiet:
        f = lambda v: "  -  " if v is None else f"{v:.3f}"
        for k in sorted(summ, key=str):
            s = summ[k]
            print(f"{k[0]:13} {k[1]} G{k[2]}  M1 {f(s['med_M1'])}  M2 {f(s['med_M2'])}  bar {f(s['med_bar'])}"
                  f"  complete {s['complete']}  failed {s['failed']}  success {s['success']:.2f}"
                  f" [{s['success_ci'][0]:.2f},{s['success_ci'][1]:.2f}]")
        for kind in ("failed_runs", "duplicates"):
            for line in problems.get(kind, []):
                print(f"{kind}: {line}")
        print("main (G384):", prim["main"], "| prior test:", prim["prior"], "| usable:", prim["usable"])
        print("win_B:", prim["win_B"])
        print("headline:", headline)
    out = dict(headline=headline, by_criterion=res,
               problems={k: sorted(map(str, v)) for k, v in problems.items()},
               summary={"%s_%s_%s" % k: v for k, v in summ.items()})
    if write:
        os.makedirs(os.path.dirname(FINAL_A2), exist_ok=True)
        json.dump(out, open(os.path.join(os.path.dirname(FINAL_A2), "report_a2.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
