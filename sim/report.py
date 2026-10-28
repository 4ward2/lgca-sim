"""Decision table from runs/final/*/test.json (brief_A.md sections 5, 9, 10, 11)."""
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(os.path.dirname(HERE), "runs", "final")
NAME = re.compile(r"^(?P<task>[PAM])_(?P<arm>[RFC])_G(?P<G>\d+)_.*?s(?P<seed>\d+)$")
EXPECTED = {"P": 10, "A": 10, "M": 5}  # amendment v2
CRITERIA = {"primary": ("M1", "M2", "bar"), "v2": ("M1_v2", "M2_v2", "bar_v2"), "ws": ("M1_ws", "M2_ws", "bar_ws"), "wT4": ("M1_wT4", "M2_wT4", "bar_wT4"), "wTmax": ("M1_wTmax", "M2_wTmax", "bar_wTmax")}
COLS = ("M1", "M2", "bar", "M2_soft", "M2_v2", "M2_ws", "M2_wT4", "M2_wTmax", "M2_fixed", "rho_halt", "rho_conv")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def load(final_dir):
    rows = defaultdict(list)
    if not os.path.isdir(final_dir):
        return rows
    for d in sorted(os.listdir(final_dir)):
        m = NAME.match(d)
        f = os.path.join(final_dir, d, "test.json")
        if not m or not os.path.exists(f):
            continue
        r = json.load(open(f))
        r["seed"] = int(m["seed"])
        fj = os.path.join(final_dir, d, "final.json")
        r["val_score"] = json.load(open(fj))["val"].get("score") if os.path.exists(fj) else None
        r["plateau"] = json.load(open(fj)).get("plateau") if os.path.exists(fj) else None
        rows[(m["task"], m["arm"], int(m["G"]))].append(r)
    return rows


def med(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return float(np.median(v)) if v else None


def summarize(rows):
    out = {}
    for (t, arm, G), rs in rows.items():
        s = {k: med(rs, k) for k in COLS + ("M1_v2", "bar_v2", "M1_ws", "bar_ws", "M1_wT4", "bar_wT4", "M1_wTmax", "bar_wTmax")}
        ks = sum((r.get("M1") or 0) >= 0.9 for r in rs)
        s["success_ci"] = wilson(ks, len(rs))
        s["outcome_frac_extra"] = [float(np.median([r["outcome_frac_extra"][i] for r in rs])) for i in range(3)]             if all("outcome_frac_extra" in r for r in rs) else None
        s.update(n=len(rs), seeds=sorted(r["seed"] for r in rs), not_plateaued=sum(r.get("plateau") is False for r in rs),
                 success=float(np.mean([(r.get("M1") or 0) >= 0.9 for r in rs])),
                 eff=float(np.median([r["M4"]["effective"] for r in rs])) if all("M4" in r for r in rs) else None)
        vs = [r for r in rs if r.get("val_score") is not None]
        if vs:
            sel = max(vs, key=lambda r: (r["val_score"], -r["seed"]))
            s["valsel_seed"], s["valsel_M2"] = sel["seed"], sel.get("M2")
        out[(t, arm, G)] = s
    return out


def verdict(out, kM1, kM2, kbar):
    """brief_A.md section 5 applied to one criterion. out: {(task, arm, G): summary}."""
    gs = sorted({G for (t, arm, G) in out if arm == "R"})
    if not gs:
        return "no R results"
    if all(out[k].get(kM1) is None for k in out if k[1] == "R"):
        return "n/a"  # criterion not present in these results
    R = lambda t, G: out.get((t, "R", G), {})
    positive = False
    for G in gs:
        wins = 0
        for t in "PAM":
            r, f = R(t, G), out.get((t, "F", G), {})
            if None in (r.get(kM1), r.get(kM2), f.get(kM2)):
                continue
            wins += r[kM1] >= 0.9 and r[kM2] - f[kM2] >= 0.10
        positive |= wins >= 2
    if all((R(t, G).get(kM1) or 0) < 0.9 for t in "PAM" for G in gs):
        return "training-failure negative"
    if not positive:
        return "NEGATIVE (kill line)"
    if any((R(t, G).get(kbar) or 0) >= 0.9 for t in "PAM" for G in gs):
        return "POSITIVE"
    return "weak positive (bar not met)"


def main(final_dir=None, write=True):
    final_dir = final_dir or FINAL
    rows = load(final_dir)
    missing = [f"{t}_{a}_G{G}: {len(rs)}/{EXPECTED[t]} seeds" for (t, a, G), rs in sorted(rows.items())
               if len(rs) != EXPECTED[t]]
    if missing:
        print("INCOMPLETE - verdict below is provisional:", "; ".join(missing))
    out = summarize(rows)
    f = lambda v: "     -" if v is None else f"{v:6.3f}"
    print(f"{'task':4} {'arm':3} {'G':>5} {'n':>2} " + " ".join(f"{c:>8}" for c in COLS) + "  success [95% CI]  valsel_M2")
    for key in sorted(out):
        s = out[key]
        print(f"{key[0]:4} {key[1]:3} {key[2]:5d} {s['n']:2d} " + " ".join(f"{f(s[c]):>8}" for c in COLS)
              + f"  {s['success']:7.2f} [{s['success_ci'][0]:.2f},{s['success_ci'][1]:.2f}]  {f(s.get('valsel_M2'))}"
              + f"  outcome(ok/wrong/unhalted) {s['outcome_frac_extra']}  not_plateaued {s['not_plateaued']}")
    vals = {name: verdict(out, *keys) for name, keys in CRITERIA.items()}
    for name, v in vals.items():
        print(f"{name}: {v}")
    present = {k: v for k, v in vals.items() if v != "n/a"}
    concl = vals["primary"] if len(set(present.values())) == 1 else \
        "criterion-dependent: " + ", ".join(f"{k}={v}" for k, v in present.items())
    print("conclusion:", concl)
    res = dict(summary={"%s_%s_%d" % k: v for k, v in out.items()}, verdicts=vals, conclusion=concl, incomplete=missing)
    if write:
        json.dump(res, open(os.path.join(os.path.dirname(final_dir), "report.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
