"""Sanity checks: gate truth tables, task labels, D99 values. No model training."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as tk
from logic import OPS, readout
from evaluate import run_hard
from scipy.ndimage import label

# 16 ops truth tables: op id k must equal the function whose truth table (a,b)=(0,0),(0,1),(1,0),(1,1)
# follows difflogic ordering
ref = {0:[0,0,0,0],1:[0,0,0,1],2:[0,0,1,0],3:[0,0,1,1],4:[0,1,0,0],5:[0,1,0,1],6:[0,1,1,0],7:[0,1,1,1],
       8:[1,0,0,0],9:[1,0,0,1],10:[1,0,1,0],11:[1,0,1,1],12:[1,1,0,0],13:[1,1,0,1],14:[1,1,1,0],15:[1,1,1,1]}
for k in range(16):
    c = OPS[k]
    tt = [float(c[0]+c[1]*a+c[2]*b+c[3]*a*b) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
    assert tt == ref[k], (k, tt)
print("ops ok")

rng = np.random.default_rng(0)
d = tk.gen_parity(rng, 50, 20)
assert (d["y"][:, -1] == d["x"][:, 0].sum(1) % 2).all()
for fc in (False, True):
    d = tk.gen_addition(rng, 200, 12, force_fullchain=fc)
    a = d["x"][:, 0].astype(int); b = d["x"][:, 1].astype(int); s = d["y"].astype(int)
    p = 2 ** np.arange(13)
    assert ((a @ p) + (b @ p) == s @ p).all()
    if fc: assert (d["diff"] >= 2).all()
assert len({tuple(z) for z in d["y"]}) > 50
print("addition ok; fullchain D mean", d["diff"].mean())
d = tk.gen_maze(rng, 100, 15)
for i in range(100):
    lab, _ = label(d["x"][i, 0] > 0)
    src = np.argwhere(d["x"][i, 1] > 0)[0]
    want = lab == lab[tuple(src)]
    assert (want == (d["y"][i] > 0)).all()
print("maze ok; D mean", d["diff"].mean(), "max", d["diff"].max())
for t in "PAM":
    print(t, "D99", tk.difficulty_quantile(t))
# input-wise parametrisation: hard coefs must equal OPS row of op_ids()
from logic import GateLayer
g = GateLayer(4, 64, torch.Generator().manual_seed(0), param="iw")
assert torch.equal(g.coefs("hard"), OPS[g.op_ids()])
print("iw ok")


# Evaluation oracle tests (synthetic mocks, seed 999; never the test seed 12345)
class Oracle(torch.nn.Module):
    """Prefix parity propagated one cell per step. kind: 'parity' | 'osc' (after step n, flipped
    at odd steps forever) | 'early' (readout last changes at step 4, whole state constant from step 5)."""
    def __init__(self, n, kind):
        super().__init__(); self.n, self.kind, self.t = n, kind, 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))
    def init_state(self, xv):
        self.t = 0; return torch.zeros((9,) + tuple(xv.shape[1:]))
    def step(self, h, xv, mode):
        self.t += 1; t = self.t; p = torch.cumsum(xv[0], 1) % 2; z = torch.zeros_like(h)
        if self.kind == "early":
            z[0] = p if t >= 4 else (p + t) % 2
            z[8].fill_(min(t, 5))
        else:
            z[0, :, :min(t, self.n)] = p[:, :min(t, self.n)]
            if self.kind == "late" and t >= 2*self.n+80: z[0] = 1 - z[0]
            if self.kind == "osc" and t > self.n and t % 2:
                z[0] = 1 - z[0]
            z[8].fill_(t if self.kind in ("osc", "late") else min(t, self.n))
        z[1:8] = z[0]  # 8 readout channels equal; channel 8 is a step counter (never read out)
        return z
    def gates_per_position(self): return 0
    def regs_per_position(self): return 0

def run_oracle(kind, n):
    x = np.random.default_rng(999).integers(0, 2, size=(50, 1, n)).astype(np.float32)
    y = (np.cumsum(x[:, 0], 1) % 2).astype(np.float32)
    return run_hard(Oracle(n, kind), "R", "P", {"x": x, "y": y}, n, m=8)

r = run_oracle("parity", 256)
print("oracle parity n=256: exact", r["correct"].mean(), "halt", r["halted"].mean(), "t_halt", r["t_halt"].mean(),
      "| v2-window m=8 exact", r["correct_m8"].mean())
assert r["correct"].mean() == 1.0 and r["halted"].mean() == 1.0
# t_halt = last readout change = position of the last 1 in the prefix-parity string (+1), <= n
assert (r["t_halt"] <= 256).all() and r["t_halt"].mean() > 240
assert r["correct_wT2"].mean() == 1.0 and r["correct_wTmax"].mean() == 1.0 and r["correct_w8"].mean() < 1.0
r = run_oracle("osc", 256)
print("oracle oscillating: exact", r["correct"].mean(), "halt", r["halted"].mean(), "v2", r["correct_v2"].mean())
assert r["correct"].mean() == 0.0 and r["halted"].mean() == 0.0 and r["correct_v2"].mean() == 1.0
for k in ("wT8", "wT4", "wT2", "wTmax"): assert r["correct_"+k].mean() == 0.0
r = run_oracle("late", 64)
print("oracle late flip:", r["correct"].mean(), r["postflip"].mean(), r["correct_wTmax"].mean())
assert r["correct"].mean() == 1.0 and r["postflip"].mean() == 1.0 and r["correct_wTmax"].mean() == 0.0
r = run_oracle("early", 16)
print("oracle early fixed point: exact", r["correct"].mean(), "halt", r["halted"].mean(), "t_halt", np.unique(r["t_halt"]))
assert r["halted"].all() and (r["t_halt"] == 4).all() and r["correct"].mean() == 1.0
print("oracle evaluation ok")

# verdict rules (brief section 5) on synthetic summaries, one case per conclusion
from report import verdict
def mk(R, F):
    """R/F: {task: (M1, M2, bar)} at G=384; same numbers for all three criteria."""
    out = {}
    for t, (m1, m2, bar) in R.items():
        out[(t, "R", 384)] = dict(M1=m1, M2=m2, bar=bar, M1_v2=m1, M2_v2=m2, bar_v2=bar, M1_ws=m1, M2_ws=m2, bar_ws=bar)
    for t, m2 in F.items():
        out[(t, "F", 384)] = dict(M2=m2, M2_v2=m2, M2_ws=m2)
    return out
F0 = {"P": 0.0, "A": 0.3, "M": 0.0}
assert verdict(mk({"P": (1, .95, .95), "A": (1, .5, .5), "M": (.95, .2, .1)}, F0), "M1", "M2", "bar") == "POSITIVE"
assert verdict(mk({"P": (1, .5, .5), "A": (1, .5, .5), "M": (.95, .2, .1)}, F0), "M1", "M2", "bar") == "weak positive (bar not met)"
assert verdict(mk({"P": (1, .95, .95), "A": (1, .35, .3), "M": (.5, .9, .9)}, F0), "M1", "M2", "bar") == "NEGATIVE (kill line)"
assert verdict(mk({"P": (.5, .9, .9), "A": (.8, .9, .9), "M": (.1, .9, .9)}, F0), "M1", "M2", "bar") == "training-failure negative"
mixed = mk({"P": (1, .95, .95), "A": (1, .5, .5), "M": (.95, .2, .1)}, F0)
for k in mixed:
    if k[1] == "R" and k[0] != "P":
        mixed[k]["M2_ws"] = 0.0
assert verdict(mixed, "M1_ws", "M2_ws", "bar_ws") == "NEGATIVE (kill line)"
only_t4 = mk({"P": (1, .95, .95), "A": (1, .5, .5), "M": (.95, .2, .1)}, F0)
for k, v in only_t4.items():
    for c in ("M1", "M2", "bar"):
        if c in v: v[c + "_wT4"] = v[c]; v[c + "_wTmax"] = v[c]
for k in only_t4:
    if k[1] == "R":
        only_t4[k]["M2_wT4"] = 0.0
assert verdict(only_t4, "M1_wT4", "M2_wT4", "bar_wT4") == "NEGATIVE (kill line)"
assert verdict(only_t4, "M1_wTmax", "M2_wTmax", "bar_wTmax") == "POSITIVE"
assert verdict({k: {kk: vv for kk, vv in v.items() if "wT" not in kk} for k, v in only_t4.items()}, "M1_wT4", "M2_wT4", "bar_wT4") == "n/a"
import tempfile, json as _json, report as _report
with tempfile.TemporaryDirectory() as td:
    fd = os.path.join(td, "final")
    for (t, arm, G), v in mixed.items():
        for seed in range(1, {"P": 11, "A": 11, "M": 6}[t]):
            d = os.path.join(fd, f"{t}_{arm}_G{G}_s{seed}"); os.makedirs(d)
            _json.dump({**v, "M4": {"effective": 1}}, open(os.path.join(d, "test.json"), "w"))
    res = _report.main(fd, write=False)
assert res["conclusion"].startswith("criterion-dependent") and not res["incomplete"], res["conclusion"]
assert res["verdicts"]["wT4"] == "n/a" and "wT4" not in res["conclusion"], res["conclusion"]
with tempfile.TemporaryDirectory() as td:
    fd = os.path.join(td, "final")
    for (t, arm, G), v in only_t4.items():
        for seed in range(1, {"P": 11, "A": 11, "M": 6}[t]):
            d = os.path.join(fd, f"{t}_{arm}_G{G}_s{seed}"); os.makedirs(d)
            _json.dump({**v, "M4": {"effective": 1}}, open(os.path.join(d, "test.json"), "w"))
    res = _report.main(fd, write=False)
assert res["conclusion"] == "criterion-dependent: primary=POSITIVE, v2=POSITIVE, ws=POSITIVE, wT4=NEGATIVE (kill line), wTmax=POSITIVE", res["conclusion"]
print("verdict tests ok")

# kaggle job builder smoke tests (isolated temporary trees)
import subprocess, tempfile, pathlib
with tempfile.TemporaryDirectory() as td:
    mf = os.path.join(td, "tune.jsonl")
    py = sys.executable
    subprocess.check_call([py, os.path.join(os.path.dirname(__file__), "kaggle_jobs.py"), "tune", "--task", "M", "--G", "384", "--out", mf, "--allow-unlocked"])
    rows = [json.loads(x) for x in open(mf, encoding="utf-8")]
    assert len(rows) == 8 and len({r["name"] for r in rows}) == 8
    assert all("sim/train.py" in r["command"] and "--task M" in r["command"] and "--out out/" in r["command"] for r in rows)
print("kaggle tune manifest ok")

