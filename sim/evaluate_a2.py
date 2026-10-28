"""Stage A2 masked evaluation for arms R/F/C (brief_A2.md).

run_masked() is sim/evaluate.run_hard with exactly two changes:
  * readout change (halting) and correctness are computed over the scoring mask only;
  * T_max / halting window come from the number of dimensions (1D: 4L, 2D: n*n), not the task name.
With an all-ones mask it reproduces run_hard bit for bit (tested in sim/test_evaluate_a2.py).
sim/evaluate.py is locked for Stage A and is not modified.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_a2 as t2
from logic import to_internal, readout, effective_gates
from evaluate import summarize, VAL2_SEED

TEST_SEED_A2 = 24680  # brief_A2 §4; only used by eval_test (locked path)
VAL_SEED = 777
N_VAL = 500
N_TEST = 500


def t_max_dims(dims, L):
    return 4 * L if dims == 1 else L * L


@torch.no_grad()
def run_masked(model, arm, dims, data, n, T_fixed=None, bs=100, m=1, mode="hard"):
    B = len(data["y"])
    res = {k: [] for k in ("correct", "bitacc", "t_conv", "converged", "correct_fixed", "t_out",
                           "correct_ws", "t_halt", "halted")}
    Tmax = t_max_dims(dims, n)
    MH = -(-Tmax // 2)
    dev = next(model.parameters()).device
    for s in range(0, B, bs):
        xv = to_internal(data["x"][s:s + bs], dev)
        y = torch.as_tensor(data["y"][s:s + bs], device=dev)
        mk = torch.as_tensor(data["mask"][s:s + bs], device=dev)
        b = y.shape[0]
        flat_m = mk.reshape(b, -1)
        allm = lambda z: (z.reshape(b, -1) | ~flat_m).all(1)      # all masked cells true
        anym = lambda z: (z.reshape(b, -1) & flat_m).any(1)       # any masked cell true
        if arm == "R":
            h = model.init_state(xv)
            tconv = torch.full((b,), -1, dtype=torch.long, device=dev)
            out_fixed = None
            t_out = torch.zeros(b, dtype=torch.long, device=dev)
            windows = {"w8": 8, "wT8": -(-Tmax // 8), "wT4": -(-Tmax // 4), "wT2": MH}
            streaks = {k: torch.zeros(b, dtype=torch.long, device=dev) for k in windows}
            ths = {k: torch.full((b,), -1, dtype=torch.long, device=dev) for k in windows}
            outs = {k: torch.zeros((b,) + tuple(y.shape[1:]), device=dev) for k in windows}
            t_halt = torch.full((b,), -1, dtype=torch.long, device=dev)
            out_halt = torch.zeros((b,) + tuple(y.shape[1:]), device=dev)
            for t in range(1, Tmax + 1):
                hn = model.step(h, xv, mode)
                same = (hn == h).reshape(hn.shape[0], b, -1).all(2).all(0)
                o_new = readout(hn, m) > 0.5
                ochg = anym(o_new != (readout(h, m) > 0.5))
                t_out[ochg] = t
                for k, w in windows.items():
                    streaks[k] = torch.where(ochg, torch.zeros_like(streaks[k]), streaks[k] + 1)
                    hnow = (ths[k] < 0) & (streaks[k] >= w)
                    ths[k][hnow] = t - w
                    outs[k][hnow] = o_new[hnow].float()
                hnow = (t_halt < 0) & (streaks["wT2"] >= MH)
                t_halt[hnow] = t - MH
                out_halt[hnow] = o_new[hnow].float()
                newly = same & (tconv < 0)
                tconv[newly] = t - 1
                h = hn
                if T_fixed is not None and t == T_fixed:
                    out_fixed = readout(h, m).clone()
                if (tconv >= 0).all() and (T_fixed is None or t >= T_fixed):
                    break
            if T_fixed is not None and out_fixed is None:
                out_fixed = readout(h, m).clone()
            out_ws = readout(h, m)
            halted = t_halt >= 0
            fp_early = ~halted & (tconv >= 0)
            t_halt[fp_early] = t_out[fp_early]
            halted = halted | fp_early
            shp = (b,) + (1,) * (y.dim() - 1)
            out = torch.where(halted.reshape(shp), out_halt, (out_ws > 0.5).float())
            out = torch.where(fp_early.reshape(shp), (out_ws > 0.5).float(), out)
            t_halt = torch.where(halted, t_halt, torch.full_like(t_halt, Tmax))
            res["t_halt"].append(t_halt); res["halted"].append(halted)
            for k, w in windows.items():
                hk = (ths[k] >= 0) | (tconv >= 0)
                okout = torch.where((ths[k] >= 0).reshape(shp), outs[k], (out_ws > 0.5).float())
                res.setdefault("correct_" + k, []).append(hk & allm((okout > 0.5) == (y > 0.5)))
            res.setdefault("correct_m8", []).append(res["correct_w8"][-1])
            res.setdefault("correct_wTmax", []).append(allm((out_ws > 0.5) == (y > 0.5)) & ((Tmax - t_out) >= MH))
            res["correct_ws"].append(allm((out_ws > 0.5) == (y > 0.5)))
            conv = tconv >= 0
            tconv = torch.where(conv, tconv, torch.full_like(tconv, Tmax))
            res["t_out"].append(t_out)
        else:
            out = readout(model(xv, mode=mode), m)
            conv = torch.ones(b, dtype=torch.bool, device=dev)
            tconv = torch.full((b,), len(model.blocks), dtype=torch.long, device=dev)
            out_fixed = None
        ok = (out > 0.5) == (y > 0.5)
        v2 = allm(ok)
        res.setdefault("correct_v2", []).append(v2)
        primary = v2 & res["halted"][-1] if arm == "R" else v2
        res["correct"].append(primary)
        if arm == "R":
            res.setdefault("postflip", []).append(res["halted"][-1] & (res["t_out"][-1] > res["t_halt"][-1]))
            res.setdefault("outcome", []).append(torch.where(res["halted"][-1], torch.where(
                primary, torch.zeros_like(primary, dtype=torch.long), torch.ones_like(primary, dtype=torch.long)),
                torch.full_like(primary, 2, dtype=torch.long)))
        else:
            for k in ("w8", "wT8", "wT4", "wT2", "wTmax"):
                res.setdefault("correct_" + k, []).append(primary)
            res.setdefault("postflip", []).append(torch.zeros_like(primary, dtype=torch.bool))
            res.setdefault("outcome", []).append(torch.where(primary, torch.zeros_like(primary, dtype=torch.long),
                                                             torch.ones_like(primary, dtype=torch.long)))
        okm = (ok.reshape(b, -1) & flat_m).float().sum(1) / flat_m.float().sum(1).clamp(min=1)
        res["bitacc"].append(okm)
        res["t_conv"].append(tconv)
        res["converged"].append(conv)
        if out_fixed is not None:
            res["correct_fixed"].append(allm((out_fixed > 0.5) == (y > 0.5)))
    return {k: torch.cat(v).cpu().numpy() for k, v in res.items() if v}


# ---- task plumbing for the new A2 tasks ----

def size_kw(task, rng=None, spec=None):
    """Training: sample (n, kw) from the train range. Evaluation: spec is an entry of TASKS_A2 indist/extra."""
    info = t2.TASKS_A2[task]
    if spec is not None:
        n = spec[0]
        if task == "inpaint":
            return n, dict(d=spec[1])
        if task == "midframe":
            return n, dict(smax=spec[1])
        return n, {}
    lo, hi = info["train_n"]
    n = int(rng.integers(lo, hi + 1))
    if task == "inpaint":
        return n, dict(d=int(rng.integers(info["train_d"][0], info["train_d"][1] + 1)))
    if task == "midframe":
        return n, dict(smax=info["train_s"][1])
    return n, {}


# validation specs (brief_A2 §3.5): in-distribution multi-size
VAL_SPECS = {
    "cycles": [(8,), (10,), (12,)], "cycles_reach": [(8,), (10,), (12,)],
    "inpaint": [(16, 2), (16, 3), (16, 4)], "midframe": [(12, 2), (14, 2), (16, 2)],
    "life": [(8,), (12,), (16,)],
}


def fixed_set_a2(task, spec, count, seed):
    n, kw = size_kw(task, spec=spec)
    rng = np.random.default_rng([seed] + [int(v) for v in spec])
    return n, t2.generate(task, rng, count, n, **kw)


def eval_val_a2(model, cfg, seed=None):
    task = cfg["task"]
    per = {}
    for spec in VAL_SPECS[task]:
        n, d = fixed_set_a2(task, spec, N_VAL, VAL_SEED if seed is None else seed)
        r = run_masked(model, cfg["arm"], 2, d, n, cfg.get("T_fixed"), m=cfg.get("readout", 1))
        per["_".join(map(str, spec))] = summarize(r, d["diff"])
    score = float(np.mean([v["exact"] for v in per.values()]))
    out = dict(score=score, per_size=per)
    out.update(per["_".join(map(str, VAL_SPECS[task][-1]))])
    return out


def eval_test_a2(model, cfg):
    """Locked path (brief_A2 §4: test seed 24680). Never called during pilots."""
    from scipy.stats import spearmanr
    task = cfg["task"]
    info = t2.TASKS_A2[task]
    specs = [info["indist"]] + list(info["extra"])
    per, pool_t, pool_d, pool_h, pool_ok = {}, [], [], [], []
    for spec in specs:
        n, d = fixed_set_a2(task, spec, N_TEST, TEST_SEED_A2)
        r = run_masked(model, cfg["arm"], 2, d, n, cfg.get("T_fixed"), m=cfg.get("readout", 1))
        key = "_".join(map(str, spec))
        per[key] = summarize(r, d["diff"])
        if "edge" in d:  # inpaint: stratify by edge-crossing vs single-colour holes (brief_A2 §5)
            for lab, sel in (("edge", d["edge"]), ("noedge", ~d["edge"])):
                if sel.any():
                    per[key]["exact_" + lab] = float(r["correct"][sel].mean())
                    per[key]["bitacc_" + lab] = float(r["bitacc"][sel].mean())
        if "t_halt" in r:
            pool_t.append(r["t_halt"]); pool_h.append(r["halted"])
        pool_d.append(d["diff"]); pool_ok.append(r["correct"])
        print(f"  {key}: {per[key]}", flush=True)
    ex = ["_".join(map(str, s)) for s in info["extra"]]
    ind = "_".join(map(str, info["indist"]))
    bar = {"cycles": "32", "cycles_reach": "32", "inpaint": "32_16", "midframe": "32_8", "life": "64"}[task]
    out = dict(hook=False, test_seed=TEST_SEED_A2, n_test=N_TEST, per_size=per, M1=per[ind]["exact"], M2=float(np.mean([per[k]["exact"] for k in ex])), bar=per[bar]["exact"])
    for suf, key in (("_v2", "exact_v2"), ("_ws", "exact_ws")):
        out["M1" + suf] = per[ind].get(key, per[ind]["exact"])
        out["M2" + suf] = float(np.mean([per[k].get(key, per[k]["exact"]) for k in ex]))
        out["bar" + suf] = per[bar].get(key, per[bar]["exact"])
    for w in ("w8", "wT8", "wT4", "wT2", "wTmax"):
        out["M1_" + w] = per[ind].get("exact_" + w, per[ind]["exact"])
        out["M2_" + w] = float(np.mean([per[k].get("exact_" + w, per[k]["exact"]) for k in ex]))
        out["bar_" + w] = per[bar].get("exact_" + w, per[bar]["exact"])
    out["outcome_frac_extra"] = [float(np.mean([per[k].get("outcome_frac", [0, 0, 0])[i] for k in ex])) for i in range(3)]
    if pool_t:
        th, hh, dd = np.concatenate(pool_t), np.concatenate(pool_h), np.concatenate(pool_d)
        out["rho_halt"] = float(spearmanr(th[hh], dd[hh])[0]) if hh.sum() > 2 and np.unique(dd[hh]).size > 1 else None
    out["M4"] = dict(G=model.gates_per_position(), effective=effective_gates(model), regs=model.regs_per_position())
    return out


def load_model_a2(run_dir):
    from logic import LoopNet, FFNet
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    info = t2.TASKS_A2[cfg["task"]]
    kw = dict(param=cfg.get("param", "softmax"), xbias=cfg.get("xbias", 0.0), trim=cfg.get("trim", False),
              wiring=cfg.get("wiring", "resid"), lconn=cfg.get("lconn", 0))
    if cfg["arm"] == "R":
        m = LoopNet(info["dims"], info["c"], cfg["G"], cfg["seed"], **kw)
    else:
        m = FFNet(info["dims"], info["c"], cfg["K"], cfg["w"], cfg["seed"], **kw)
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    m.eval()
    return m, cfg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--locked", action="store_true", help="evaluate the A2 test sets (seed 24680)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m, cfg = load_model_a2(a.run_dir)
    if not a.locked:
        print(json.dumps(eval_val_a2(m, cfg), indent=1))
    else:
        res = eval_test_a2(m, cfg)
        json.dump(res, open(os.path.join(a.run_dir, "test.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "per_size"}, indent=1))
