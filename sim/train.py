"""Train one run (brief_A.md section 3). Only in-distribution validation is evaluated here."""
import argparse
import json
import math
import os
import sys
import time
import hashlib, subprocess

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as tk
from logic import LoopNet, FFNet, to_internal, readout
from evaluate import eval_val

D99_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d99.json")


def d99(task):
    cache = json.load(open(D99_CACHE)) if os.path.exists(D99_CACHE) else {}
    if task not in cache:
        cache[task] = tk.difficulty_quantile(task)
        json.dump(cache, open(D99_CACHE, "w"), indent=1)
    return cache[task]


PERCH = False  # per-channel BCE over the m readout channels (Fable diagnosis: the mean objective destroys correct solutions)


def bce(h, y, m=1):
    if PERCH:
        p = (1e-4 + (1 - 2e-4) * h[:m]).clamp(1e-4, 1 - 1e-4)
        if not torch.isfinite(p).all():
            return torch.tensor(float("nan"), device=p.device)
        return F.binary_cross_entropy(p, y.unsqueeze(0).expand_as(p))
    p = 1e-4 + (1 - 2e-4) * readout(h, m)  # linear squeeze: keeps gradients when h is exactly 0/1 (st mode)
    p = p.clamp(1e-4, 1 - 1e-4)  # guards float drift only; in-range values keep their gradient
    if not torch.isfinite(p).all():
        return torch.tensor(float("nan"), device=p.device)
    return F.binary_cross_entropy(p, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(tk.TASKS))
    ap.add_argument("--arm", required=True, choices=["R", "F", "C"])
    ap.add_argument("--G", type=int, default=384)
    ap.add_argument("--K", type=int, default=1, help="F: number of blocks")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--resid", type=float, default=0.0)
    ap.add_argument("--mode", default="soft", choices=["soft", "st"])
    ap.add_argument("--param", default="softmax", choices=["softmax", "iw"])
    ap.add_argument("--readout", type=int, default=1, help="majority over first m state channels")
    ap.add_argument("--trim", action="store_true", help="forbid constant gates")
    ap.add_argument("--sched", default="const", choices=["const", "cos"])
    ap.add_argument("--st_frac", type=float, default=0.0, help="last fraction of steps trained with hard forward (st)")
    ap.add_argument("--keep_best", action="store_true", help="save the checkpoint with best in-dist val exact")
    ap.add_argument("--opt", default="adam", choices=["adam", "dlca"],
                    help="dlca: AdamW(b2=0.99, wd=1e-2) + grad clip by value 100, as in the DiffLogic CA notebook")
    ap.add_argument("--wiring", default="resid", choices=["resid", "nbr"])
    ap.add_argument("--perch", action="store_true", help="per-channel BCE over the readout channels")
    ap.add_argument("--lconn", type=int, default=0, help="learnable B-input candidates per gate (0 = fixed)")
    ap.add_argument("--xbias", type=float, default=0.0, help="layer-1 B-input drawn from task inputs w.p. xbias")
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    global PERCH
    PERCH = a.perch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = tk.TASKS[a.task]
    dims, c = info["dims"], info["c"]
    bs = a.bs or (128 if dims == 1 else 32)
    D99 = d99(a.task)
    if a.arm == "R":
        model = LoopNet(dims, c, a.G, a.seed, a.resid, a.param, a.xbias, a.trim, a.wiring, a.lconn)
        w, K = a.G // 3, 1
    elif a.arm == "F":
        assert a.G % (3 * a.K) == 0
        w, K = a.G // (3 * a.K), a.K
        model = FFNet(dims, c, K, w, a.seed, a.resid, a.param, a.xbias, a.trim, a.wiring, a.lconn)
    else:
        w, K = a.G // 3, int(math.ceil(1.5 * D99))
        model = FFNet(dims, c, K, w, a.seed, a.resid, a.param, a.xbias, a.trim, a.wiring, a.lconn)
    model.to(dev)
    T_fixed = int(math.ceil((1 + a.beta) / 2 * D99)) if a.arm == "R" else None
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(vars(a), w=w, K=K, D99=D99, T_fixed=T_fixed, bs=bs,
               gates_per_position=model.gates_per_position())
    try: cfg["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: cfg["git_commit"] = None
    cfg["sha256"] = {f: hashlib.sha256(open(os.path.join(os.path.dirname(__file__), f), "rb").read()).hexdigest() for f in ("tasks.py","evaluate.py","logic.py")}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    if a.opt == "dlca":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    if a.sched == "cos":
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))
    else:
        sch = None
    rng = np.random.default_rng([a.seed, 1000])
    best_score, best_step = -1.0, 0
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    t0 = time.time()
    run_loss = None
    bad_steps = 0
    best = None
    for step in range(1, a.steps + 1):
        mode = "st" if (a.st_frac > 0 and step > (1 - a.st_frac) * a.steps) else a.mode
        n = tk.train_size(a.task, rng)
        d = tk.generate(a.task, rng, bs, n)
        xv = to_internal(d["x"], dev)
        y = torch.as_tensor(d["y"], device=dev)
        if a.arm == "R":
            Db = int(d["diff"].max())
            T = int(math.ceil(rng.uniform(1, a.beta) * Db))
            loss = 0.0
            if a.alpha < 1:
                loss = loss + (1 - a.alpha) * bce(model(xv, T, mode, a.ckpt), y, a.readout)
            if a.alpha > 0:
                n1 = int(rng.integers(0, T))
                k = int(rng.integers(max(1, Db - n1), T - n1 + 1))
                with torch.no_grad():
                    h = model(xv, n1, mode)
                loss = loss + a.alpha * bce(model(xv, k, mode, a.ckpt, h=h.detach()), y, a.readout)
        else:
            loss = bce(model(xv, mode=mode, ckpt=a.ckpt), y, a.readout)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            bad_steps += 1
            continue
        loss.backward()
        if a.opt == "dlca":
            torch.nn.utils.clip_grad_value_(model.parameters(), 100.0)
        opt.step()
        if sch is not None:
            sch.step()
        lv = loss.item()
        run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
        if step % 50 == 0:
            rec = dict(step=step, loss=round(run_loss, 5), sec=round(time.time() - t0, 1), bad=bad_steps)
            if step % a.eval_every == 0 or step == a.steps:
                model.eval()
                cfg_eval = dict(cfg)
                rec["val"] = eval_val(model, cfg_eval)
                model.train()
                score = rec["val"]["score"] + 1e-3 * rec["val"]["bitacc"]
                if a.keep_best and score > best_score:
                    best_score, best_step = score, step
                    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               os.path.join(a.out, "model.pt"))
                rec["best_step"] = best_step
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            print(f"KG_PROGRESS {step}/{a.steps}", flush=True)
    if not a.keep_best or best_step == 0:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "model.pt"))
    else:
        model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location="cpu"))
        model.to(dev)
    model.eval()
    final = eval_val(model, cfg); val2 = eval_val(model, cfg, seed=778)
    vals = []
    try:
        for line in open(os.path.join(a.out, "log.jsonl")):
            z=json.loads(line)
            if "val" in z: vals.append((z["step"], z["val"]["score"]))
    except Exception: pass
    peak_all=max([v for _,v in vals]+[final["score"]]); peak_early=max([v for s,v in vals if s <= .75*a.steps]+[0.0])
    json.dump(dict(val=final, val2=val2, best_score=best_score, sec=round(time.time() - t0, 1), best_step=best_step,
                   plateau=bool(peak_all-peak_early <= .05), plateau_best=peak_all, plateau_early_best=peak_early),
              open(os.path.join(a.out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
