"""A5 item 5: delayed length-generalisation probe of arm R (Fable review section 3; design_A5_v2.md item 5).

Stage A R recipe (sim/common.json + the per-task final configs in runs/final/*/config.json), imported from
sim/logic.py (LoopNet), sim/tasks.py (generators), sim/train.py (bce, d99), sim/evaluate.py (run_hard, summarize):
  G=384, wiring nbr, lconn 4, xbias 0.5, resid 10, trim, readout over 8 channels, per-channel BCE, AdamW(0.9, 0.99),
  lr 0.1, clip-by-value 100, batch 128, soft training;  A: alpha 0, beta 4 (Stage A final A_R_G384_alpha0.0_beta4.0);
  P: alpha 0.5, beta 2 (Stage A final P_R_G384_alpha0.5_beta2.0).  Training stream rng = default_rng([seed, 1000]),
  model init seed = seed, exactly as sim/train.py, so P seeds 1/3/7 start from the same init and stream as Stage A.

Schedule (EXPLICIT; stored in config.json as `schedule`): total --steps (60000).  For t < head_steps the lr multiplier is
  head=cos : floor + (1-floor) * 0.5 * (1 + cos(pi * t / head_steps))        (Stage A's cosine, stretched over the head only)
  head=const: 1.0                                                          (what Stage A literally ran: sched const)
and for t >= head_steps it is the constant tail_frac (head=cos: tail_frac = floor, the cosine's final value;
the design's "final value" would be 0, which freezes training and makes wd inert, so floor > 0 is a stated deviation).
Weight decay: AdamW multiplies parameters by (1 - lr_t * wd) every step, so the total decay over the run is
  D = -sum_t log(1 - lr_t * wd).  wd is solved (bisection) for the targets D in {1, 3, 10, 30} (i.e. e^-1 ... e^-30)
and both wd and the realised D are written to config.json.

Log every 500 steps (log.jsonl): exact-match (halted-and-correct, sim/evaluate.run_hard) per length 16/32/64/128/256 in
hard AND soft mode on fixed dev sets (seed 300900), BCE (soft, n=16, T = T_fixed), parameter L2 norm, mean gate entropy
and mean max-probability of the 16-op softmax.  Resumable: ckpt.pt in --out every eval; --resume DIR continues,
--stop_at N ends a segment at step N (the job DAG chains segments).
Criterion (summarize_log): train n=16 hard exact >= 0.99 for >= 10k consecutive steps, then running max of n=32 hard
exact > 0.9 (n=64 secondary).
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch

from . import SIM_DIR  # noqa: F401  (puts sim/ on sys.path)
import tasks as tk  # noqa: E402
import train as st  # noqa: E402
import evaluate as ev  # noqa: E402
from logic import LoopNet, GateLayer, to_internal  # noqa: E402

DEV_SEED = 300900
LENGTHS = (16, 32, 64, 128, 256)
COUNTS = {16: 200, 32: 200, 64: 100, 128: 40, 256: 20}
RECIPE = dict(
    G=384, lr=0.1, resid=10.0, wiring="nbr", lconn=4, xbias=0.5, trim=True, readout=8, perch=True, param="softmax",
    mode="soft", clip_value=100.0, betas=(0.9, 0.99), bs=128,
    per_task={"A": dict(alpha=0.0, beta=4.0, head_steps=6000), "P": dict(alpha=0.5, beta=2.0, head_steps=4000)})


def lr_mult(t, head_steps, head, tail_frac):
    if t >= head_steps:
        return tail_frac
    if head == "const":
        return 1.0
    return tail_frac + (1 - tail_frac) * 0.5 * (1 + math.cos(math.pi * t / head_steps))


def total_decay(wd, lr, steps, head_steps, head, tail_frac):
    return -sum(math.log1p(-lr * lr_mult(t, head_steps, head, tail_frac) * wd) for t in range(steps))


def wd_for_decay(target, lr, steps, head_steps, head, tail_frac):
    lo, hi = 0.0, 0.99 / lr
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if total_decay(mid, lr, steps, head_steps, head, tail_frac) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gate_stats(model):
    ent, mx, n = 0.0, 0.0, 0
    for m in model.modules():
        if isinstance(m, GateLayer):
            p = torch.softmax(m.logits.detach() + m.mask, 1)
            ent += float(-(p * torch.log(p.clamp_min(1e-12))).sum(1).sum())
            mx += float(p.max(1)[0].sum())
            n += p.shape[0]
    return ent / n, mx / n


def param_norm(model):
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())))


def evaluate(model, cfg, task, lengths):
    out = {}
    for n in lengths:
        d = tk.fixed_set(task, n, COUNTS[n], DEV_SEED)
        for mode in ("hard", "soft"):
            r = ev.run_hard(model, "R", task, d, n, None, m=RECIPE["readout"], mode=mode)
            out[f"{mode}_{n}"] = float(r["correct"].mean())
    return out


def dev_bce(model, cfg, task, mode="soft"):
    d = tk.fixed_set(task, 16, 200, DEV_SEED)
    xv = to_internal(d["x"])
    y = torch.as_tensor(d["y"])
    T = cfg["T_fixed"]
    with torch.no_grad():
        return float(st.bce(model(xv, T, mode), y, RECIPE["readout"]))


def summarize_log(log, k_steps=10000, ood_thr=0.9):
    """Pre-registered criterion: sustained train n=16 >= 0.99 (hard) for >= k_steps, then running max of n=32 > ood_thr."""
    steps = [r["step"] for r in log]
    ok = [r["hard_16"] >= 0.99 for r in log]
    start = None
    sustained_from = None
    for i, s in enumerate(steps):
        if ok[i]:
            start = steps[i] if start is None else start
            if s - start >= k_steps and sustained_from is None:
                sustained_from = s
        else:
            start = None
    res = dict(train_sustained_reached=sustained_from is not None, sustained_from_step=sustained_from,
               max_hard_32_overall=max([r["hard_32"] for r in log] or [0]),
               max_hard_64_overall=max([r.get("hard_64", 0.0) for r in log] or [0]),
               max_soft_32_overall=max([r.get("soft_32", 0.0) for r in log] or [0]))
    if sustained_from is not None:
        after = [r for r in log if r["step"] >= sustained_from]
        res["running_max_hard_32_after"] = max(r["hard_32"] for r in after)
        res["running_max_hard_64_after"] = max(r.get("hard_64", 0.0) for r in after)
        res["flipped"] = res["running_max_hard_32_after"] > ood_thr
    else:
        res["flipped"] = False
    return res


def make_cfg(a):
    pt = RECIPE["per_task"][a.task]
    head_steps = a.head_steps or pt["head_steps"]
    tail_frac = a.tail_frac if a.tail_frac is not None else (0.1 if a.head == "cos" else 1.0)
    lr = RECIPE["lr"]
    if a.wd is not None:
        wd = a.wd
    else:
        wd = wd_for_decay(a.decay, lr, a.steps, head_steps, a.head, tail_frac) if a.decay > 0 else 0.0
    info = tk.TASKS[a.task]
    D99 = st.d99(a.task)
    T_fixed = int(math.ceil((1 + pt["beta"]) / 2 * D99))
    cfg = dict(task=a.task, seed=a.seed, steps=a.steps, wd=wd, target_total_decay_exponent=a.decay,
               realised_total_decay_exponent=total_decay(wd, lr, a.steps, head_steps, a.head, tail_frac),
               schedule=dict(head=a.head, head_steps=head_steps, tail_frac=tail_frac, lr=lr,
                             text="lr_t = lr * mult(t); head=cos: tail_frac+(1-tail_frac)*0.5*(1+cos(pi t/head_steps)) for "
                                  "t<head_steps; head=const: 1; t>=head_steps: tail_frac (constant tail)"),
               eval_every=a.eval_every, lengths=list(LENGTHS), counts=COUNTS, dev_seed=DEV_SEED, dims=info["dims"],
               c=info["c"], D99=D99, T_fixed=T_fixed, recipe={k: v for k, v in RECIPE.items() if k != "per_task"},
               task_recipe=pt, opt="AdamW", note="positive control: P seeds 1,3,7 failed in Stage A (runs/final/P_R_G384_*)")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["A", "P"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--decay", type=float, default=0.0, help="target total decay exponent x (D = e^-x); 0 = wd 0")
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--stop_at", type=int, default=None, help="end this segment at this global step")
    ap.add_argument("--head", choices=["cos", "const"], default="cos")
    ap.add_argument("--head_steps", type=int, default=None)
    ap.add_argument("--tail_frac", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--lengths", type=int, nargs="*", default=list(LENGTHS))
    ap.add_argument("--resume", default=None, help="directory containing ckpt.pt (and log.jsonl) of the previous segment")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = make_cfg(a)
    sch = cfg["schedule"]
    info = tk.TASKS[a.task]
    pt = RECIPE["per_task"][a.task]
    model = LoopNet(info["dims"], info["c"], RECIPE["G"], a.seed, RECIPE["resid"], RECIPE["param"], RECIPE["xbias"],
                    RECIPE["trim"], RECIPE["wiring"], RECIPE["lconn"])
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], betas=RECIPE["betas"], weight_decay=cfg["wd"])
    rng = np.random.default_rng([a.seed, 1000])
    st.PERCH = RECIPE["perch"]
    step0, run_loss, log = 0, None, []
    if a.resume:
        ck = torch.load(os.path.join(a.resume, "ckpt.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        step0, run_loss = ck["step"], ck["run_loss"]
        with open(os.path.join(a.resume, "config.json"), encoding="utf-8") as f:
            old = json.load(f)
        for k in ("task", "seed", "steps", "wd", "schedule"):
            assert old[k] == cfg[k], f"resume config mismatch on {k}: {old[k]} vs {cfg[k]}"
        with open(os.path.join(a.resume, "log.jsonl"), encoding="utf-8") as f:
            log = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    stop = min(a.stop_at or a.steps, a.steps)
    logf = open(os.path.join(a.out, "log.jsonl"), "w", encoding="utf-8")
    for r in log:
        logf.write(json.dumps(r) + "\n")
    t0 = time.time()
    bad = 0
    for step in range(step0 + 1, stop + 1):
        lr = RECIPE["lr"] * lr_mult(step - 1, sch["head_steps"], sch["head"], sch["tail_frac"])
        for g in opt.param_groups:
            g["lr"] = lr
        n = tk.train_size(a.task, rng)
        d = tk.generate(a.task, rng, RECIPE["bs"], n)
        xv = to_internal(d["x"])
        y = torch.as_tensor(d["y"])
        Db = int(d["diff"].max())
        T = int(math.ceil(rng.uniform(1, pt["beta"]) * Db))
        loss = 0.0
        if pt["alpha"] < 1:
            loss = loss + (1 - pt["alpha"]) * st.bce(model(xv, T, "soft"), y, RECIPE["readout"])
        if pt["alpha"] > 0:
            n1 = int(rng.integers(0, T))
            k = int(rng.integers(max(1, Db - n1), T - n1 + 1))
            with torch.no_grad():
                h = model(xv, n1, "soft")
            loss = loss + pt["alpha"] * st.bce(model(xv, k, "soft", False, h=h.detach()), y, RECIPE["readout"])
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            bad += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), RECIPE["clip_value"])
        opt.step()
        lv = float(loss.detach())
        run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
        if step % a.eval_every == 0 or step == stop:
            model.eval()
            rec = dict(step=step, lr=lr, train_loss_ema=run_loss, bad_steps=bad, sec=round(time.time() - t0, 1))
            rec.update(evaluate(model, cfg, a.task, a.lengths))
            rec["bce_soft16"] = dev_bce(model, cfg, a.task, "soft")
            rec["bce_hard16"] = dev_bce(model, cfg, a.task, "hard")
            rec["param_norm"] = param_norm(model)
            rec["gate_entropy"], rec["gate_maxprob"] = gate_stats(model)
            model.train()
            log.append(rec)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state, step=step,
                            run_loss=run_loss), os.path.join(a.out, "ckpt.pt"))
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    logf.close()
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(task=a.task, seed=a.seed, wd=cfg["wd"], reached_step=stop, complete=stop == a.steps,
                       criterion=summarize_log(log) if all("hard_16" in r and "hard_32" in r for r in log) else None),
                  f, indent=1)


if __name__ == "__main__":
    main()
