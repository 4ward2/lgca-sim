"""A5 arm-B trainer: BPNet with latent size K, soft sum-product BP, A2 recipe, deployed with rounding_k.

  python -m sim.a5.train_b --task {m1d,m2d_dense,m2d_last,delay3,delay4} --K 4 --seed 300101 --out out/<job>

Recipe (b_arm.py / brief_A2 section 3): Adam lr 0.03, batch 128 (1-D) / 32 (2-D), T = ceil(u * D), u ~ U(1, 2),
2000 steps, grad value clip 10, masked BCE, evaluate every 250 steps on a fixed dev set (best_deploy = deployed dev
score, ties -> earlier step).  D = batch['diff'], except M2D where D = 3n (the K=4 oracle needs <= 3n rounds; the
carry-run 'diff' of tasks_mul would give far too few BP rounds).  Deployment = rounding_k.round_bp on ONE
training-stream calibration batch seeded by (seed, 9000 + step), evaluated with the ac.py interpreter.

Metric layers (design item 6): `exact_fp`  = readout at the fixed point is exact on the mask;
                               `exact_halt` = halted-and-correct (state at a fixed point, or readout unchanged for the
                               last ceil(T_max/2) rounds, and exact at T_max).  T_max = 4L (chains), 4n (n x 2n grids).
Sizes: m1d train n 2..6, dev 4/5/6, test 8/12/16 (+ the family a = 2^(n-1), bound 2^(K+2-n));
       m2d train n 2..4, dev 3/4, test 5/6/8;  delay train length 8..16, dev 8/12/16, test 32/64/128.
Seeds: model/training stream = --seed; dev set seed 300777; test set seed 300778 (both >= 300000).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from . import ac, rounding_k as rk, tasks_delay as td, tasks_mul as tm  # noqa: F401  (puts sim/ on sys.path)
import b_arm as B  # noqa: E402
from logic import to_internal  # noqa: E402

DEV_SEED, TEST_SEED = 300777, 300778
LR, STEPS, EVAL_EVERY = 0.03, 2000, 250

TASKS = {  # name: kind, c, dims, train sizes (inclusive), dev sizes, test sizes
    "m1d": dict(kind="M1D", c=2, dims=1, train=(2, 6), dev=(4, 5, 6), test=(8, 12, 16)),
    "m2d_dense": dict(kind="M2D", c=2, dims=2, train=(2, 4), dev=(3, 4), test=(5, 6, 8), mask="mask_all"),
    "m2d_last": dict(kind="M2D", c=2, dims=2, train=(2, 4), dev=(3, 4), test=(5, 6, 8), mask="mask"),
    "delay3": dict(kind="DELAY", d=3, c=1, dims=1, train=(8, 16), dev=(8, 12, 16), test=(32, 64, 128)),
    "delay4": dict(kind="DELAY", d=4, c=1, dims=1, train=(8, 16), dev=(8, 12, 16), test=(32, 64, 128)),
}


def dump(obj, path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)


def gen(task, rng, B_, n):
    """One batch of size B_ at size parameter n (M1D/M2D: n bits; delay: chain length n). Keys: x y mask diff."""
    t = TASKS[task]
    if t["kind"] == "DELAY":
        return td.generate(rng, B_, n, t["d"])
    d = tm.generate(t["kind"], rng, B_, n)
    if t["kind"] == "M2D":
        d["mask"] = d[t["mask"]]
        d["diff"] = np.full(B_, 3 * n, np.int64)
    return d


def fixed_set(task, n, count, seed):
    return gen(task, np.random.default_rng([seed, n, 17]), count, n)


def train_batch(task, rng, B_):
    lo, hi = TASKS[task]["train"]
    return gen(task, rng, B_, int(rng.integers(lo, hi + 1)))


def batch_size(task):
    return 128 if TASKS[task]["dims"] == 1 else 32


def t_max(x_shape):
    return rk.default_tmax(x_shape)


def run_ac(tabs, d, tmax=None):
    """Two-layer evaluation of a deployed table set on a data dict. Returns per-sample bool arrays."""
    u, r, o = tabs["u"], tabs["r"], tabs["o"]
    xi = ac.codes_from_x(d["x"])
    y = d["y"] > 0.5
    mk = d["mask"]
    Tmax = tmax or t_max(d["x"].shape)
    MH = -(-Tmax // 2)
    cap = 4 * Tmax
    K = u.shape[1]
    c = np.ones(xi.shape + (K,), bool)
    b = xi.shape[0]
    flat = lambda a: a.reshape(b, -1)
    prev = ac.readout(o, c, xi)
    t_out = np.zeros(b, np.int64)
    fp_at = np.full(b, -1, np.int64)
    out_T = None
    if Tmax == 0:
        out_T = prev
    for t in range(1, cap + 1):
        cn = ac.update(u, r, c, xi)
        same = flat(cn == c).all(1)
        fp_at[(fp_at < 0) & same] = t - 1
        c = cn
        cur = ac.readout(o, c, xi)
        t_out[flat(cur != prev).any(1)] = t
        prev = cur
        if t == Tmax:
            out_T = cur.copy()
        if (fp_at >= 0).all() and t >= Tmax:
            break
    if out_T is None:
        out_T = prev
    fp_conv = fp_at >= 0
    ok = lambda out: (flat(((out > 0) == y) | ~mk)).all(1)
    ok_T, ok_fp = ok(out_T), ok(prev)
    halted = (fp_conv & (fp_at <= Tmax)) | ((Tmax - t_out) >= MH)
    return dict(exact_fp=ok_fp, exact_halt=ok_T & halted, fp_conv=fp_conv, t_fp=np.where(fp_conv, fp_at, cap),
                halted=halted, exact_T=ok_T)


def summarize(res):
    return dict(exact_fp=float(res["exact_fp"].mean()), exact_halt=float(res["exact_halt"].mean()),
                exact_at_Tmax=float(res["exact_T"].mean()), fp_conv_rate=float(res["fp_conv"].mean()),
                halted_rate=float(res["halted"].mean()), mean_t_fp=float(res["t_fp"].mean()))


def eval_sizes(task, tabs, sizes, count, seed):
    per = {}
    for n in sizes:
        d = fixed_set(task, n, count, seed)
        per[str(n)] = summarize(run_ac(tabs, d))
    return per


def eval_family(tabs, K, sizes, count, seed):
    """M1D a = 2^(n-1) family: exact-match must not exceed 2^(K+2-n) (Fable theorem 2)."""
    out = {}
    for n in sizes:
        rng = np.random.default_rng([seed, n, 99])
        Bv = rng.integers(0, 1 << n, size=count)
        d = tm._from_ab("M1D", [1 << (n - 1)] * count, Bv, n)
        s = summarize(run_ac(tabs, d))
        s["bound_2^(K+2-n)"] = min(1.0, 2.0 ** (K + 2 - n))
        out[str(n)] = s
    return out


def deploy(model, task, seed, step):
    rng = np.random.default_rng([seed, 9000 + step])
    fit = train_batch(task, rng, batch_size(task))
    tabs, nseen = rk.round_bp(model, fit, t_max(fit["x"].shape))
    return tabs, nseen


def tables_json(tabs, K):
    j = rk.to_json(tabs)
    j["K"] = K
    return j


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eval_every", type=int, default=EVAL_EVERY)
    ap.add_argument("--n_dev", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    assert a.seed >= 300000 and a.seed not in (12345, 24680)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t = TASKS[a.task]
    model = B.BPNet(t["dims"], t["c"], K=a.K, seed=a.seed)
    rng = np.random.default_rng(a.seed + 1000)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bs = batch_size(a.task)
    cfg = dict(arm="B", task=a.task, K=a.K, seed=a.seed, steps=a.steps, eval_every=a.eval_every, lr=LR, batch=bs,
               beta=2.0, thr=rk.THR, dev_seed=DEV_SEED, test_seed=TEST_SEED, n_dev=a.n_dev, n_test=a.n_test,
               train_sizes=t["train"], dev_sizes=t["dev"], test_sizes=t["test"], grad_clip_value=10.0,
               T_rule="ceil(u*D), u~U(1,2); D=diff (M2D: 3n)", Tmax_rule="4L chain / 4n grid",
               git_commit=B.git_commit())
    dump(cfg, os.path.join(a.out, "config.json"), indent=1)
    log, best, best_tabs, best_step = [], -1.0, None, 0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        d = train_batch(a.task, rng, bs)
        xv = to_internal(d["x"])
        y = torch.as_tensor(d["y"])
        mk = torch.as_tensor(d["mask"])
        T = max(1, int(math.ceil(float(rng.uniform(1.0, 2.0)) * int(d["diff"].max()))))
        p, _, _ = model(xv, T)
        loss = B.masked_bce(p, y, mk)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                tabs, nseen = deploy(model, a.task, a.seed, step)
                per = eval_sizes(a.task, tabs, t["dev"], a.n_dev, DEV_SEED)
            model.train()
            score = float(np.mean([v["exact_halt"] for v in per.values()]))
            log.append(dict(step=step, loss=float(loss.detach()), T=T, sec=round(time.time() - t0, 1),
                            dev_score=score, dev_per=per, n_obs_pairs=nseen))
            print(json.dumps({k: v for k, v in log[-1].items() if k != "dev_per"}), flush=True)
            if score > best:  # ties -> earlier step
                best, best_tabs, best_step = score, tabs, step
    dump(log, os.path.join(a.out, "log.json"), indent=1)
    dev = eval_sizes(a.task, best_tabs, t["dev"], a.n_dev, DEV_SEED)
    dump(dict(task=a.task, K=a.K, seed=a.seed, best_deploy_step=best_step, dev_score=best, dev_per=dev,
                   sec=round(time.time() - t0, 1)), os.path.join(a.out, "final.json"), indent=1)
    test = dict(task=a.task, K=a.K, seed=a.seed, best_deploy_step=best_step,
                per_size=eval_sizes(a.task, best_tabs, t["test"], a.n_test, TEST_SEED))
    if a.task == "m1d":
        test["family_a_2^(n-1)"] = eval_family(best_tabs, a.K, t["test"], a.n_test, TEST_SEED)
    dump(test, os.path.join(a.out, "test.json"), indent=1)
    dump(tables_json(best_tabs, a.K), os.path.join(a.out, "tables.json"))
    print("DONE", json.dumps(dict(task=a.task, K=a.K, seed=a.seed, best_step=best_step, dev=best,
                                  test={k: v["exact_halt"] for k, v in test["per_size"].items()})), flush=True)


if __name__ == "__main__":
    main()
