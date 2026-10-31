"""Validate sim/a5/jobs_a5.json.  python -m sim.a5.dag_check [path]

Checks: schema keys, unique names, needs refer to existing jobs, acyclic, every command writes only to out/<own name>/
(no other out/ path), every in/ path in a command is in/<own name>/<one of its needs>/, B jobs have disjoint seed sets
>= 300000 (never 12345/24680), formal B needs the pilot of the same (task, K), R segment chain is contiguous,
report needs every B job and every final R segment.
"""
import json
import os
import re
import sys


def check(path):
    errs = []
    data = json.load(open(path, encoding="utf-8"))
    jobs = data["jobs"]
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        errs.append("duplicate job names")
    by = {j["name"]: j for j in jobs}
    for j in jobs:
        for k in ("name", "needs", "command"):
            if k not in j:
                errs.append(f"{j.get('name')}: missing {k}")
        for n in j["needs"]:
            if n not in by:
                errs.append(f"{j['name']}: unknown need {n}")
        for m in re.finditer(r"\bout/([\w.-]+)", j["command"]):
            if m.group(1) != j["name"]:
                errs.append(f"{j['name']}: writes to out/{m.group(1)}")
        for m in re.finditer(r"\bin/([\w.-]+)/([\w.-]+)", j["command"]):
            if m.group(1) != j["name"] or m.group(2) not in j["needs"]:
                errs.append(f"{j['name']}: reads in/{m.group(1)}/{m.group(2)} not in needs")
    state = {}

    def visit(n):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            errs.append(f"cycle through {n}")
            return
        state[n] = 1
        for m in by.get(n, {}).get("needs", []):
            visit(m)
        state[n] = 2

    for n in names:
        visit(n)
    seen = {}
    for j in jobs:
        if j.get("kind") == "B":
            for s in j["seeds"]:
                if s < 300000 or s in (12345, 24680):
                    errs.append(f"{j['name']}: bad seed {s}")
                key = (j["task"], j["K"], j["formal"])
                if s in seen.setdefault(key, set()):
                    errs.append(f"{j['name']}: duplicate seed {s}")
                seen[key].add(s)
            if j["formal"]:
                p = f"b-{j['task']}-k{j['K']}-pilot"
                if p not in j["needs"]:
                    errs.append(f"{j['name']}: formal job does not need {p}")
                if set(j["seeds"]) & set(by[p]["seeds"]):
                    errs.append(f"{j['name']}: formal seeds overlap pilot")
    finals = []
    for j in jobs:
        if j.get("kind") == "R":
            if j["segment"] == 1:
                if j["needs"]:
                    errs.append(f"{j['name']}: first segment has needs")
            else:
                exp = re.sub(r"-g\d+$", f"-g{j['segment'] - 1}", j["name"])
                if j["needs"] != [exp]:
                    errs.append(f"{j['name']}: segment chain broken")
            if j["segment"] == j["segments"]:
                finals.append(j["name"])
    rep = by.get("report")
    if rep is None:
        errs.append("no report job")
    else:
        want = {j["name"] for j in jobs if j.get("kind") == "B"} | set(finals)
        if set(rep["needs"]) != want:
            errs.append("report needs mismatch")
    return errs, len(jobs)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs_a5.json")
    e, n = check(p)
    print(f"{n} jobs;", "OK" if not e else f"{len(e)} errors")
    for x in e[:50]:
        print(" -", x)
    sys.exit(1 if e else 0)
