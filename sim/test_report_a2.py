"""Synthetic tests for report_a2 (every branch of brief_A2 §6, plus the F1/F4/F9 run-validity rules)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_a2 as ra

CRIT_KEYS = ("", "_v2", "_ws", "_wT4", "_wTmax")


def write(base, task, arm, G, seed, M1, M2, bar=None, fail=False, crit=None, raw=None, cfg=None):
    name = f"{task}_{arm}_G{G}_cfg_s{seed}" if arm != "B" else f"{task}_B_s{seed}"
    d = os.path.join(base, name)
    os.makedirs(d)
    json.dump({}, open(os.path.join(d, "final.json"), "w"))
    json.dump(cfg or {"seed": seed, "steps": 2000, "eval_every": 250}, open(os.path.join(d, "config.json"), "w"))
    if fail:
        return
    if raw is not None:
        open(os.path.join(d, "test.json"), "w").write(raw)
        return
    r = {}
    for k in CRIT_KEYS:
        m1, m2 = (crit or {}).get(k, (M1, M2))
        r["M1" + k], r["M2" + k], r["bar" + k] = m1, m2, m2 if bar is None else bar
    json.dump(r, open(os.path.join(d, "test.json"), "w"))


def dirs():
    td = tempfile.mkdtemp()
    fa, fa2 = os.path.join(td, "A"), os.path.join(td, "A2")
    os.makedirs(fa); os.makedirs(fa2)
    return fa, fa2


def scenario(B, F, R=None, skip=(), fail_seed=None, critB=None):
    fa, fa2 = dirs()
    R = R or {t: (0.0, 0.0) for t in F}
    for t in F:
        base_rf = fa if t in ra.MAIN else fa2
        for G in (384, 1536):
            for s in ra.SEEDS:
                write(base_rf, t, "F", G, s, *F[t])
                write(base_rf, t, "R", G, s, *R[t])
    for t in B:
        for s in ra.SEEDS:
            if (t, s) in skip:
                continue
            write(fa2, t, "B", None, s, *B[t], fail=(fail_seed == (t, s)), crit=(critB or {}).get(t))
    return ra.main(fa, fa2, write=False, quiet=True)


ALL = ra.MAIN + ra.PRIOR + ra.NEW
BTASKS = tuple(t for t in ALL if t != "midframe")
F0 = {t: (0.5, 0.0) for t in ALL}
good, bad = (1.0, 1.0), (0.2, 0.05)
P = lambda r: r["by_criterion"]["primary"]["G384"]

r = scenario({t: good for t in BTASKS}, F0)
assert P(r)["main"] == "POSITIVE" and P(r)["prior"] == "rule-learning positive" and P(r)["usable"], P(r)
assert r["headline"] == "POSITIVE"
assert P(r)["win_B"]["midframe"].startswith("N/A")            # F9: not run, not a protocol failure

r = scenario({"P": good, "A": good, "M": good, "cycles_reach": good, "cycles": bad, "inpaint": bad, "life": bad}, F0)
assert P(r)["prior"] == "prior-matched positive"
r = scenario({"P": good, "A": good, "M": good, "cycles_reach": bad, "cycles": bad, "inpaint": bad, "life": bad}, F0)
assert P(r)["prior"].startswith("Q control invalid")
r = scenario({"P": good, "A": bad, "M": bad, **{t: good for t in ra.PRIOR + ("inpaint", "life")}}, F0)
assert P(r)["main"] == "NEGATIVE"
r = scenario({"P": good, "A": good, "M": bad, **{t: bad for t in ra.PRIOR + ("inpaint", "life")}}, F0)
assert P(r)["main"] == "POSITIVE"

# equality counts as a win (0.6 - 0.5 is 0.0999... in floating point)
r = scenario({"P": (0.9, 0.6), "A": (0.9, 0.6), "M": bad, **{t: bad for t in ra.PRIOR + ("inpaint", "life")}},
             {**F0, "P": (0.5, 0.5), "A": (0.5, 0.5)})
assert P(r)["win_B"]["P"] is True

# one failed run counts as 0 in the median; a missing seed -> protocol failure
r = scenario({t: good for t in BTASKS}, F0, fail_seed=("A", 3))
assert P(r)["win_B"]["A"] is True
r = scenario({t: good for t in BTASKS}, F0, skip=(("M", 4),))
assert P(r)["main"].startswith("protocol failure")

# criterion dependence
r = scenario({t: good for t in BTASKS}, F0, critB={"A": {"_wT4": (1.0, 0.0)}, "M": {"_wT4": (1.0, 0.0)}})
assert r["headline"].startswith("criterion-dependent") and "wT4=NEGATIVE" in r["headline"], r["headline"]

# F1: NaN / Infinity / missing required field / bad JSON -> whole run failed (0), never a win
for raw in ('{"M1": 1.0, "M2": 1.0, "bar": NaN}', '{"M1": 1.0, "M2": Infinity, "bar": 1.0}',
            '{"M1": 1.0, "M2": 1.0}', "not json", "[1, 2]"):
    fa, fa2 = dirs()
    for G in (384, 1536):
        for s in ra.SEEDS:
            write(fa, "P", "F", G, s, 0.5, 0.0); write(fa, "P", "R", G, s, 0.0, 0.0)
    for s in ra.SEEDS:
        write(fa2, "P", "B", None, s, 0, 0, raw=raw)
    probs = {}
    summ = ra.summarize(ra.collect(fa, fa2, probs))
    b = summ[("P", "B", None)]
    assert b["failed"] == 10 and b["med_M1"] == 0.0 and b["med_M2"] == 0.0, (raw, b)
    assert ra.win(summ, "P", "B") is False and len(probs["failed_runs"]) == 10

# F4: duplicate seed directories -> protocol failure for that (task, arm, G)
fa, fa2 = dirs()
for G in (384, 1536):
    for s in ra.SEEDS:
        write(fa, "P", "F", G, s, 0.5, 0.0); write(fa, "P", "R", G, s, 0.0, 0.0)
for s in ra.SEEDS:
    write(fa2, "P", "B", None, s, 1.0, 1.0)
d1 = os.path.join(fa, "P_F_G384_K8_s1"); os.makedirs(d1)
json.dump({}, open(os.path.join(d1, "final.json"), "w"))
json.dump({"M1": 0.9, "M2": 0.9, "bar": 0.9}, open(os.path.join(d1, "test.json"), "w"))
probs = {}
rows = ra.collect(fa, fa2, probs)
assert ("P", "F", 384) in probs["dup_keys"]
assert ra.win(ra.summarize(rows, dup_keys=probs["dup_keys"]), "P", "B") is None

# F4: B with a smoke budget or a wrong seed in config is rejected
for cfg, why in (({"seed": 1, "steps": 100, "eval_every": 50}, "budget"), ({"seed": 7, "steps": 2000, "eval_every": 250}, "seed")):
    fa, fa2 = dirs()
    write(fa2, "P", "B", None, 1, 1.0, 1.0, cfg=cfg)
    probs = {}
    ra.collect(fa, fa2, probs)
    assert any(why in x for x in probs["failed_runs"]), probs

# secondary criterion missing in valid runs -> 'n/a', not a scientific negative
fa, fa2 = dirs()
for G in (384, 1536):
    for s in ra.SEEDS:
        write(fa, "P", "F", G, s, 0.5, 0.0)
for s in ra.SEEDS:
    d = os.path.join(fa2, f"P_B_s{s}"); os.makedirs(d)
    json.dump({}, open(os.path.join(d, "final.json"), "w"))
    json.dump({"seed": s, "steps": 2000, "eval_every": 250}, open(os.path.join(d, "config.json"), "w"))
    json.dump({"M1": 1.0, "M2": 1.0, "bar": 1.0}, open(os.path.join(d, "test.json"), "w"))
summ = ra.summarize(ra.collect(fa, fa2, {}), suffix="_wT4")
assert ra.win(summ, "P", "B") == "n/a"
print("report_a2 tests ok")
