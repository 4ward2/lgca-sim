"""Hard-gate evaluation (brief_A.md section 4).

Validation (in-distribution, seed 777) may be run at any time.
The extrapolation test sets (seed 12345) are only evaluated with --locked, i.e. in the
locked run after the common settings have been written to NOTES.md.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as tk
from logic import LoopNet, FFNet, to_internal, effective_gates, readout


M8 = 8
VAL2_SEED = 778


def m_halt(task, n):
    """amendment v4 (brief_A.md section 11): readout must be unchanged for ceil(T_max/2) steps."""
    return -(-t_max_for(task, n) // 2)


def t_max_for(task, n):
    if tk.TASKS[task]["dims"] == 1:
        L = n + 1 if task == "A" else n
        return 4 * L
    return n * n


@torch.no_grad()
def run_hard(model, arm, task, data, n, T_fixed=None, bs=None, m=1, mode="hard"):
    """Returns per-sample dict: correct, bitacc, t_conv, converged, and correct_fixed (R only)."""
    B = len(data["y"])
    if bs is None:
        bs = 200 if tk.TASKS[task]["dims"] == 1 else 100
    res = {k: [] for k in ("correct", "bitacc", "t_conv", "converged", "correct_fixed", "t_out",
                           "correct_ws", "t_halt", "halted")}
    Tmax = t_max_for(task, n)
    dev = next(model.parameters()).device
    for s in range(0, B, bs):
        xv = to_internal(data["x"][s:s + bs], dev)
        y = torch.as_tensor(data["y"][s:s + bs], device=dev)
        b = y.shape[0]
        sp_dims = tuple(range(1, y.dim()))
        if arm == "R":
            h = model.init_state(xv)
            tconv = torch.full((b,), -1, dtype=torch.long, device=dev)
            out_fixed = None
            t_out = torch.zeros(b, dtype=torch.long, device=dev)  # exploratory: last step the output bits changed
            # amendment v4: halt when the readout bits are unchanged for MH consecutive steps (v2 used 8)
            MH = m_halt(task, n)
            windows = {"w8": 8, "wT8": -(-Tmax//8), "wT4": -(-Tmax//4), "wT2": MH}
            streaks = {k: torch.zeros(b, dtype=torch.long, device=dev) for k in windows}
            ths = {k: torch.full((b,), -1, dtype=torch.long, device=dev) for k in windows}
            outs = {k: torch.zeros((b,) + tuple(y.shape[1:]), device=dev) for k in windows}
            t_halt = torch.full((b,), -1, dtype=torch.long, device=dev)
            out_halt = torch.zeros((b,) + tuple(y.shape[1:]), device=dev)
            for t in range(1, Tmax + 1):
                hn = model.step(h, xv, mode)
                same = (hn == h).reshape(hn.shape[0], b, -1).all(2).all(0)
                o_new = readout(hn, m) > 0.5
                ochg = (o_new != (readout(h, m) > 0.5)).reshape(b, -1).any(1)
                t_out[ochg] = t
                for k,w in windows.items():
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
                    break  # whole-state fixed point everywhere: readout can no longer change
            if T_fixed is not None and out_fixed is None:
                out_fixed = readout(h, m).clone()  # converged before T_fixed: fixed point
            out_ws = readout(h, m)  # original criterion: state at whole-state fixed point / T_max
            halted = t_halt >= 0
            nh = ~halted
            # whole-state fixed point implies the readout is stable from then on
            fp_early = nh & (tconv >= 0)
            # the readout cannot change after a fixed point: it halted at its last change (t_out)
            t_halt[fp_early] = t_out[fp_early]
            halted = halted | fp_early
            out = torch.where(halted.reshape((b,) + (1,) * (y.dim() - 1)), out_halt, (out_ws > 0.5).float())
            out = torch.where(fp_early.reshape((b,) + (1,) * (y.dim() - 1)), (out_ws > 0.5).float(), out)
            t_halt = torch.where(halted, t_halt, torch.full_like(t_halt, Tmax))
            res["t_halt"].append(t_halt); res["halted"].append(halted)
            for k,w in windows.items():
                hk = (ths[k] >= 0) | (tconv >= 0)
                okout = torch.where((ths[k] >= 0).reshape((b,) + (1,)*(y.dim()-1)), outs[k], (out_ws>0.5).float())
                res.setdefault("correct_"+k, []).append(hk & ((okout>0.5)==(y>0.5)).reshape(b,-1).all(1))
            res.setdefault("correct_m8", []).append(res["correct_w8"][-1])
            # wTmax: correct at T_max and no change in the last ceil(T_max/2) steps.
            res.setdefault("correct_wTmax", []).append((((out_ws>0.5)==(y>0.5)).reshape(b,-1).all(1) & ((Tmax-t_out)>=MH)))
            okw = (out_ws > 0.5) == (y > 0.5)
            res["correct_ws"].append(okw.reshape(b, -1).all(1))
            conv = tconv >= 0
            tconv = torch.where(conv, tconv, torch.full_like(tconv, Tmax))
            res["t_out"].append(t_out)
        else:
            out = readout(model(xv, mode=mode), m)
            conv = torch.ones(b, dtype=torch.bool, device=dev)
            tconv = torch.full((b,), len(model.blocks), dtype=torch.long, device=dev)
            out_fixed = None
        pred = out > 0.5
        ok = pred == (y > 0.5)
        v2 = ok.reshape(b, -1).all(1)
        res.setdefault("correct_v2", []).append(v2)
        primary = v2
        if arm == "R":
            primary = primary & res["halted"][-1]
        res["correct"].append(primary)
        if arm == "R":
            res.setdefault("postflip", []).append(res["halted"][-1] & (res["t_out"][-1] > res["t_halt"][-1]))
            res.setdefault("outcome", []).append(torch.where(res["halted"][-1], torch.where(primary, torch.zeros_like(primary), torch.ones_like(primary)), torch.full_like(primary, 2)))
        else:
            for k in ("w8","wT8","wT4","wT2","wTmax"): res.setdefault("correct_"+k, []).append(primary)
            res.setdefault("postflip", []).append(torch.zeros_like(primary, dtype=torch.bool)); res.setdefault("outcome", []).append(torch.where(primary, torch.zeros_like(primary), torch.ones_like(primary)))
        res["bitacc"].append(ok.reshape(b, -1).float().mean(1))
        res["t_conv"].append(tconv)
        res["converged"].append(conv)
        if out_fixed is not None:
            okf = (out_fixed > 0.5) == (y > 0.5)
            res["correct_fixed"].append(okf.reshape(b, -1).all(1))
    return {k: torch.cat(v).cpu().numpy() for k, v in res.items() if v}


def summarize(r, diff):
    out = dict(exact=float(r["correct"].mean()), exact_v2=float(r.get("correct_v2", r["correct"]).mean()), bitacc=float(r["bitacc"].mean()),
               conv_rate=float(r["converged"].mean()),
               mean_t=float(r["t_conv"].mean()),
               conv_wrong=float((r["converged"] & ~r["correct"]).mean()))
    if "correct_fixed" in r:
        out["exact_fixed"] = float(r["correct_fixed"].mean())
    if "t_out" in r:
        out["mean_t_out"] = float(r["t_out"].mean())
    if "correct_m8" in r:
        out["exact_m8"] = float(r["correct_m8"].mean())
    if "t_halt" in r:
        out["halt_rate"] = float(r["halted"].mean())
        out["mean_t_halt"] = float(r["t_halt"].mean())
        out["exact_ws"] = float(r["correct_ws"].mean())
    for k in ("w8","wT8","wT4","wT2","wTmax"):
        if "correct_"+k in r: out["exact_"+k] = float(r["correct_"+k].mean())
    if "postflip" in r: out["postflip_rate"] = float(r["postflip"].mean())
    if "outcome" in r: out["outcome_frac"] = [float((r["outcome"]==i).mean()) for i in range(3)]
    out["mean_D"] = float(np.mean(diff))
    return out


def load_model(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    c = tk.TASKS[cfg["task"]]["c"]
    dims = tk.TASKS[cfg["task"]]["dims"]
    if cfg["arm"] == "R":
        m = LoopNet(dims, c, cfg["G"], cfg["seed"], param=cfg.get("param", "softmax"), xbias=cfg.get("xbias", 0.0), trim=cfg.get("trim", False), wiring=cfg.get("wiring", "resid"), lconn=cfg.get("lconn", 0))
    else:
        m = FFNet(dims, c, cfg["K"], cfg["w"], cfg["seed"], param=cfg.get("param", "softmax"), xbias=cfg.get("xbias", 0.0), trim=cfg.get("trim", False), wiring=cfg.get("wiring", "resid"), lconn=cfg.get("lconn", 0))
    m.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    m.eval()
    return m, cfg


def val_set(task, n):
    return tk.fixed_set(task, n, tk.N_VAL[task], tk.VAL_SEED)


def eval_val(model, cfg, seed=None):
    task = cfg["task"]
    sizes = {"P": (8,12,16), "A": (8,12,16), "M": (8,10,12)}[task]
    per = {}
    for n in sizes:
        d = tk.fixed_set(task, n, tk.N_VAL[task], tk.VAL_SEED if seed is None else seed)
        r = run_hard(model, cfg["arm"], task, d, n, cfg.get("T_fixed"), m=cfg.get("readout",1))
        per[n] = summarize(r, d["diff"])
    score = float(np.mean([per[n]["exact"] for n in sizes]))
    out = dict(score=score, per_size={str(n):v for n,v in per.items()})
    out.update(per[sizes[-1]])
    return out

def eval_test(model, cfg):
    from scipy.stats import spearmanr
    task = cfg["task"]
    info = tk.TASKS[task]
    sizes = (info["indist"],) + tuple(info["extra"])
    per = {}
    pool_t, pool_d, pool_ok, pool_conv, pool_to, pool_th, pool_h = [], [], [], [], [], [], []
    for n in sizes:
        kw = {"fullchain_p": 0.0} if task == "A" else {}
        d = tk.fixed_set(task, n, tk.N_TEST[task], tk.TEST_SEED, **kw)
        t0 = time.time()
        r = run_hard(model, cfg["arm"], task, d, n, cfg.get("T_fixed"), m=cfg.get("readout", 1))
        per[n] = summarize(r, d["diff"])
        per[n]["sec"] = round(time.time() - t0, 1)
        pool_t.append(r["t_conv"]); pool_d.append(d["diff"])
        pool_ok.append(r["correct"]); pool_conv.append(r["converged"])
        if "t_out" in r:
            pool_to.append(r["t_out"])
        if "t_halt" in r:
            pool_th.append(r["t_halt"]); pool_h.append(r["halted"])
        print(f"  n={n}: {per[n]}", flush=True)
        if task == "A":
            ds = tk.fixed_set(task, n, tk.N_TEST[task], tk.TEST_SEED, force_fullchain=True)
            rs = run_hard(model, cfg["arm"], task, ds, n, cfg.get("T_fixed"), m=cfg.get("readout", 1))
            per[n]["stress"] = summarize(rs, ds["diff"])
            per[n]["bins_stress"] = {k: float(rs["correct"][((ds["diff"] >= lo) & (ds["diff"] <= hi))].mean()) if ((ds["diff"] >= lo) & (ds["diff"] <= hi)).any() else None for k,lo,hi in (("1-4",1,4),("5-8",5,8),("9-16",9,16),("17-32",17,32),("33+",33,10**9))}
            per[n]["bins_uniform"] = {k: float(r["correct"][((d["diff"] >= lo) & (d["diff"] <= hi))].mean()) if ((d["diff"] >= lo) & (d["diff"] <= hi)).any() else None for k,lo,hi in (("1-4",1,4),("5-8",5,8),("9-16",9,16),("17-32",17,32),("33+",33,10**9))}
            print(f"  n={n} stress: {per[n]['stress']}", flush=True)
    out = dict(per_size={str(k): v for k, v in per.items()},
               M1=per[info["indist"]]["exact"],
               M2=float(np.mean([per[n]["exact"] for n in info["extra"]])),
               bar=per[info["bar"]]["exact"],
               M1_v2=per[info["indist"]]["exact_v2"],
               M2_v2=float(np.mean([per[n]["exact_v2"] for n in info["extra"]])),
               bar_v2=per[info["bar"]]["exact_v2"],
               M1_ws=per[info["indist"]].get("exact_ws", per[info["indist"]]["exact"]),
               M2_ws=float(np.mean([per[n].get("exact_ws", per[n]["exact"]) for n in info["extra"]])),
               bar_ws=per[info["bar"]].get("exact_ws", per[info["bar"]]["exact"]))
    for w in ("w8","wT8","wT4","wT2","wTmax"):
        out["M1_"+w] = per[info["indist"]].get("exact_"+w, per[info["indist"]]["exact"])
        out["M2_"+w] = float(np.mean([per[n].get("exact_"+w, per[n]["exact"]) for n in info["extra"]]))
        out["bar_"+w] = per[info["bar"]].get("exact_"+w, per[info["bar"]]["exact"])
    out["outcome_frac_extra"] = [float(np.mean([per[n].get("outcome_frac", [0,0,0])[i] for n in info["extra"]])) for i in range(3)]
    if cfg["arm"] == "R":
        soft = []
        for n in info["extra"]:
            dsoft = tk.fixed_set(task, n, tk.N_TEST[task], tk.TEST_SEED, **({"fullchain_p":0.0} if task == "A" else {}))
            soft.append(summarize(run_hard(model, cfg["arm"], task, dsoft, n, cfg.get("T_fixed"), m=cfg.get("readout",1), mode="soft"), dsoft["diff"])["exact"])
        out["M2_soft"] = float(np.mean(soft))
    if cfg["arm"] == "R":
        out["M2_fixed"] = float(np.mean([per[n]["exact_fixed"] for n in info["extra"]]))
        t = np.concatenate(pool_t); dd = np.concatenate(pool_d)
        ok = np.concatenate(pool_ok); cv = np.concatenate(pool_conv)
        out["rho_conv"] = float(spearmanr(t[cv], dd[cv])[0]) if cv.sum() > 2 else None
        sel = cv & ok
        out["rho_correct"] = float(spearmanr(t[sel], dd[sel])[0]) if sel.sum() > 2 else None
        # amendment v2 primary M3: halting step (readout stable for M_HALT steps) vs difficulty
        th = np.concatenate(pool_th); hh = np.concatenate(pool_h)
        out["rho_halt"] = float(spearmanr(th[hh], dd[hh])[0]) if hh.sum() > 2 else None
        out["rho_halt_correct"] = float(spearmanr(th[hh & ok], dd[hh & ok])[0]) if (hh & ok).sum() > 2 else None
        out["M2_ws"] = float(np.mean([per[n]["exact_ws"] for n in info["extra"]]))
        # exploratory, not pre-registered: output-bits-only stability
        to = np.concatenate(pool_to)
        out["explor_rho_t_out"] = float(spearmanr(to, dd)[0])
        out["explor_rho_t_out_correct"] = float(spearmanr(to[ok], dd[ok])[0]) if ok.sum() > 2 else None
    largest = max(sizes)
    pos = largest if info["dims"] == 1 else largest * largest
    if task == "A":
        pos = largest + 1
    G = model.gates_per_position()
    out["M4"] = dict(G=G, effective=effective_gates(model), regs=model.regs_per_position(), control=19 if cfg.get("readout",1)==8 else 0,
                     halt_counter_bits=int(math.ceil(math.log2(m_halt(task, largest)+1))) if cfg["arm"]=="R" else 0,
                     total_gates_largest=G * pos, total_regs_largest=model.regs_per_position() * pos)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--locked", action="store_true", help="evaluate extrapolation test sets")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m, cfg = load_model(a.run_dir)
    if not a.locked:
        print(json.dumps(eval_val(m, cfg), indent=1))
    else:
        res = eval_test(m, cfg)
        json.dump(res, open(os.path.join(a.run_dir, "test.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "per_size"}, indent=1))
