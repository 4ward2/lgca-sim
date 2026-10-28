"""Synthetic tests for report_a2 (every branch of brief_A2 §6, plus the F1/F4/F9 run-validity rules)."""
import json
import os
import sys
import tempfile
import shutil
import atexit
import b_arm
import tasks
import evaluate_a2

TEMP_DIRS = []
atexit.register(lambda: [shutil.rmtree(p) for p in TEMP_DIRS])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_a2 as ra

assert ra.DEPLOY == b_arm.DEPLOY and ra.THR == b_arm.THR
assert ra.B_BUDGET == dict(steps=b_arm.STEPS, eval_every=b_arm.EVAL_EVERY)
assert ra.N_TEST_A == tasks.N_TEST and ra.N_TEST_A2 == evaluate_a2.N_TEST

CRIT_KEYS = ("", "_v2", "_ws", "_wT4", "_wTmax")


def write(base, task, arm, G, seed, M1, M2, bar=None, fail=False, crit=None, raw=None, cfg=None):
    name = f"{task}_{arm}_G{G}_cfg_s{seed}" if arm != "B" else f"{task}_B_s{seed}"
    d = os.path.join(base, name)
    os.makedirs(d)
    json.dump(dict(last_step=2000, best_deploy_step=250, val={"score": 1.0}, val2={"score": 1.0}), open(os.path.join(d, "final.json"), "w"))
    config = dict(task=task, arm=arm, seed=seed, steps=2000 if arm == "B" else 6000,
                  eval_every=250, n_val=None, deploy=ra.DEPLOY.get(task), thr=ra.THR, G=G, tuned=1)
    config.update(cfg or {})
    json.dump(config, open(os.path.join(d, "config.json"), "w"))
    if arm in "RF":
        sel = f"selected_{task}_G{G}_{arm}.json" if task in ra.MAIN else f"selected_{task}_{arm}.json"
        json.dump([dict(cfg={"tuned": 1}, median=1.0)], open(os.path.join(os.path.dirname(base), sel), "w"))
    if fail:
        return
    if raw is not None:
        open(os.path.join(d, "test.json"), "w").write(raw)
        return
    r = dict(hook=False, test_seed=12345 if task in ra.MAIN else 24680,
             n_test=ra.N_TEST_A[task] if task in ra.MAIN else ra.N_TEST_A2, checkpoint_steps={"best_deploy": 250})
    for k in CRIT_KEYS:
        m1, m2 = (crit or {}).get(k, (M1, M2))
        r["M1" + k], r["M2" + k], r["bar" + k] = m1, m2, m2 if bar is None else bar
    json.dump(r, open(os.path.join(d, "test.json"), "w"))
    if arm in "RF" and task in ra.MAIN:
        target = os.path.join(os.path.dirname(base), "inherit_stageA.json")
        records = json.load(open(target)) if os.path.exists(target) else {}
        records[os.path.abspath(d)] = ra._hashes(d)
        json.dump(records, open(target, "w"))


def dirs():
    td = tempfile.mkdtemp()
    TEMP_DIRS.append(td)
    fa, fa2 = os.path.join(td, "A"), os.path.join(td, "A2")
    os.makedirs(fa); os.makedirs(fa2)
    return fa, fa2


def scenario(B, F, R=None, skip=(), fail_seed=None, critB=None):
    fa, fa2 = dirs()
    R = R or {t: (0.0, 0.0) for t in F}
    for t in F:
        base_rf = fa if t in ra.MAIN else fa2
        for G in ((384, 1536) if t in ra.MAIN else (384,)):
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
    write(fa2, "P", "B", None, s, 1.0, 1.0)
    d = os.path.join(fa2, f"P_B_s{s}")
    r = json.load(open(os.path.join(d, "test.json")))
    for k in list(r):
        if any(k.endswith(suf) for suf in CRIT_KEYS[1:]):
            del r[k]
    json.dump(r, open(os.path.join(d, "test.json"), "w"))
summ = ra.summarize(ra.collect(fa, fa2, {}), suffix="_wT4")
assert ra.win(summ, "P", "B") == "n/a"
# Every malformed artifact must fail admission without escaping an exception.
for filename, key, value, why in (
    ("final", None, {}, "val.score"), ("final", None, [], "not an object"),
    ("final", "last_step", 1999, "last_step"), ("config", None, [], "not an object"),
    ("config", "seed", None, "seed"), ("config", "seed", True, "seed"),
    ("config", "task", "A", "task"), ("config", "arm", "F", "arm"),
    ("config", "steps", 2000.9, "steps"), ("config", "steps", "2000", "steps"),
    ("config", "eval_every", True, "eval_every"), ("config", "n_val", 50, "n_val"),
    ("config", "deploy", "wrong", "deploy"), ("config", "thr", 3, "thr"),
    ("final", "best_deploy_step", 251, "best_deploy_step"),
    ("test", "hook", True, "hook"), ("test", "test_seed", 777, "test_seed"),
    ("test", "n_test", 2, "n_test"), ("test", "checkpoint_steps", {"best_deploy": 500}, "best_deploy")):
    fa, fa2 = dirs()
    write(fa2, "P", "B", None, 1, 1, 1)
    path = os.path.join(fa2, "P_B_s1", filename + ".json")
    obj = json.load(open(path))
    if key is None:
        obj = value
    else:
        obj[key] = value
    json.dump(obj, open(path, "w"))
    rec, reason = ra._check_run(fa2, "P_B_s1", 1, "B", fa2)
    assert rec is None and why in reason, (filename, key, reason)
for stage_a in (False, True):
    for failure in (("selected cfg mismatch", "no selection record") if not stage_a else
                    ("hash mismatch", "not in inheritance list", "no inheritance list", "selected cfg mismatch", "no selection record")):
        fa, fa2 = dirs()
        base, task = (fa, "P") if stage_a else (fa2, "life")
        write(base, task, "R", 384, 1, 1, 1)
        name = f"{task}_R_G384_cfg_s1"
        target = os.path.join(os.path.dirname(base), "inherit_stageA.json")
        sel = os.path.join(os.path.dirname(base), f"selected_{task}_G384_R.json" if stage_a else f"selected_{task}_R.json")
        if failure == "hash mismatch":
            with open(os.path.join(base, name, "test.json"), "a") as f: f.write(" ")
        elif failure == "not in inheritance list": json.dump({}, open(target, "w"))
        elif failure == "no inheritance list": os.remove(target)
        elif failure == "no selection record": os.remove(sel)
        else: json.dump([dict(cfg={"tuned": 2}, median=1)], open(sel, "w"))
        rec, reason = ra._check_run(base, name, 1, "R", fa2)
        assert rec is None and failure in reason, reason
# Inheritance is additive and refuses changes atomically.
fa, fa2 = dirs()
write(fa, "P", "R", 384, 1, 1, 1)
ra.inherit(fa, fa2)
target = os.path.join(os.path.dirname(fa2), "inherit_stageA.json")
before = open(target, "rb").read()
with open(os.path.join(fa, "P_R_G384_cfg_s1", "test.json"), "a") as f: f.write(" ")
try:
    ra.inherit(fa, fa2)
    raise AssertionError("changed inherited run accepted")
except ValueError:
    assert open(target, "rb").read() == before
r = scenario({t: good for t in BTASKS}, F0)
g = r["by_criterion"]["primary"]["G1536"]
assert g["prior"] == "N/A (G1536: P/A/M only, brief_A2 §6)"
assert set(g["win_B"]) == set(ra.MAIN) and g["new_lines"] == {}
assert r["sources"] and all(os.path.isabs(p) for paths in r["sources"].values() for p in paths)
s = ra.summarize({("P", "B", None): {1: {"M1": 1, "M2": 0, "bar": 0}}}, seeds=(1,))
assert s[("P", "B", None)]["success_indist"] == 1 and s[("P", "B", None)]["success_extra"] == 0
# R1/R3: independent mutations of otherwise admitted new-task R/F runs.
for arm in ("R", "F"):
    for filename, key, value, why in (
        ("directory", None, "life_{}_G1536_cfg_s1", "directory G"),
        ("directory", None, "life_{}_cfg_s1", "directory G"),
        ("config", "G", 1536, "G mismatch"),
        ("config", "G", 384.0, "G mismatch"),
        ("config", "steps", 5999, "steps"),
        ("test", "hook", True, "hook"),
        ("test", "test_seed", 777, "test_seed"),
        ("test", "n_test", 2, "n_test")):
        fa, fa2 = dirs()
        write(fa2, "life", arm, 384, 1, 1, 1)
        name = f"life_{arm}_G384_cfg_s1"
        assert ra._check_run(fa2, name, 1, arm, fa2)[0] is not None
        if filename == "directory":
            new = value.format(arm)
            os.rename(os.path.join(fa2, name), os.path.join(fa2, new))
            name = new
        else:
            path = os.path.join(fa2, name, filename + ".json")
            obj = json.load(open(path)); obj[key] = value
            json.dump(obj, open(path, "w"))
        rec, reason = ra._check_run(fa2, name, 1, arm, fa2)
        assert rec is None and why in reason, (filename, key, reason)

# The highest median may be later; equal maxima select the first.
fa, fa2 = dirs()
write(fa2, "life", "R", 384, 1, 1, 1)
name = "life_R_G384_cfg_s1"
sel = os.path.join(os.path.dirname(fa2), "selected_life_R.json")
json.dump([dict(cfg={"tuned": 2}, median=0), dict(cfg={"tuned": 1}, median=1)], open(sel, "w"))
assert ra._check_run(fa2, name, 1, "R", fa2)[0] is not None
json.dump([dict(cfg={"tuned": 1}, median=1), dict(cfg={"tuned": 2}, median=1)], open(sel, "w"))
assert ra._check_run(fa2, name, 1, "R", fa2)[0] is not None
path = os.path.join(fa2, name, "config.json")
obj = json.load(open(path)); obj["tuned"] = 2
json.dump(obj, open(path, "w"))
rec, reason = ra._check_run(fa2, name, 1, "R", fa2)
assert rec is None and "selected cfg mismatch" in reason

# Merge a genuinely unlisted complete Y, skip incomplete Z, refuse deleted X atomically.
fa, fa2 = dirs()
write(fa, "P", "R", 384, 1, 1, 1)
target = os.path.join(os.path.dirname(fa2), "inherit_stageA.json")
x = os.path.abspath(os.path.join(fa, "P_R_G384_cfg_s1"))
original = json.load(open(target))[x]
write(fa, "A", "R", 384, 1, 1, 1)
y = os.path.abspath(os.path.join(fa, "A_R_G384_cfg_s1"))
# write() normally registers Stage A fixtures; remove Y to exercise the merge.
json.dump({x: original}, open(target, "w"))
write(fa, "M", "R", 384, 1, 1, 1, fail=True)
z = os.path.abspath(os.path.join(fa, "M_R_G384_cfg_s1"))
merged = ra.inherit(fa, fa2)
assert set(merged) == {x, y} and merged[x] == original and z not in merged
assert json.load(open(target)) == merged
before = open(target, "rb").read()
shutil.rmtree(x)
try:
    ra.inherit(fa, fa2)
    raise AssertionError("deleted inherited run accepted")
except ValueError as exc:
    assert "listed directory missing" in str(exc)
    assert open(target, "rb").read() == before

# Producer identity: synthetic data/scoring only, with real generators forbidden.
from unittest.mock import Mock, patch
import numpy as np
calls = []
def fake_fixed(task, spec, count, seed):
    calls.append((task, spec, count, seed))
    return spec[0], {"diff": np.array([1, 2])}
def fake_run(*args, **kwargs):
    return {"correct": np.array([True, False])}
with patch.object(evaluate_a2, "fixed_set_a2", side_effect=fake_fixed), \
        patch.object(evaluate_a2.t2, "generate", side_effect=AssertionError("real generation")), \
        patch.object(evaluate_a2, "run_masked", side_effect=fake_run), \
        patch.object(evaluate_a2, "summarize", return_value={"exact": 0.5}), \
        patch.object(evaluate_a2, "effective_gates", return_value=1):
    model = Mock()
    model.gates_per_position.return_value = 1
    model.regs_per_position.return_value = 1
    produced = evaluate_a2.eval_test_a2(model, dict(task="life", arm="R"))
assert produced["hook"] is False
assert produced["test_seed"] == evaluate_a2.TEST_SEED_A2 and produced["n_test"] == evaluate_a2.N_TEST
assert calls and len(calls) == 1 + len(evaluate_a2.t2.TASKS_A2["life"]["extra"])
assert all(seed == evaluate_a2.TEST_SEED_A2 and count == evaluate_a2.N_TEST for _, _, count, seed in calls)

for missing in ("cycles", "cycles_reach"):
    r = scenario({t: good for t in BTASKS if t != missing}, F0)
    assert P(r)["main"] == "POSITIVE"
    assert P(r)["prior"] == "protocol failure (Q/Q+ incomplete)", P(r)
print("report_a2 tests ok")
