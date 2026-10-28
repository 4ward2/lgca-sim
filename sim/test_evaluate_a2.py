"""run_masked must equal evaluate.run_hard bit for bit with an all-ones mask; masking must ignore
unmasked cells. Uses validation-seed data only."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as tk
import evaluate as ev
import evaluate_a2 as ea
torch.set_num_threads(3)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cases = [("P", 12, os.path.join(ROOT, "runs", "final", "P_R_G384_alpha0.5_beta2.0_s2")),
         ("P", 16, os.path.join(ROOT, "runs", "final", "P_F_G384_K4_s1")),
         ("M", 10, os.path.join(ROOT, "runs", "final", "M_F_G384_K8_s1"))]
mruns = sorted(d for d in os.listdir(os.path.join(ROOT, "runs", "final")) if d.startswith("M_R_G384"))
if mruns:
    cases.append(("M", 10, os.path.join(ROOT, "runs", "final", mruns[0])))
for task, n, rd in cases:
    model, cfg = ev.load_model(rd)
    d = tk.fixed_set(task, n, 100, tk.VAL_SEED)
    d["mask"] = np.ones_like(d["y"], dtype=bool)
    a = ev.run_hard(model, cfg["arm"], task, d, n, cfg.get("T_fixed"), m=cfg.get("readout", 1))
    b = ea.run_masked(model, cfg["arm"], tk.TASKS[task]["dims"], d, n, cfg.get("T_fixed"),
                      bs=200 if tk.TASKS[task]["dims"] == 1 else 100, m=cfg.get("readout", 1))
    for k in a:
        assert np.array_equal(a[k], b[k]), (task, cfg["arm"], k)
    print("identical:", task, cfg["arm"], n, "exact", a["correct"].mean())
    # masking: flipping labels outside a mask must not change the masked score
    rng = np.random.default_rng(0)
    mk = rng.random(d["y"].shape) < 0.3
    d2 = dict(d, mask=mk)
    d3 = dict(d2, y=np.where(mk, d["y"], 1 - d["y"]))
    r2 = ea.run_masked(model, cfg["arm"], tk.TASKS[task]["dims"], d2, n, cfg.get("T_fixed"), m=cfg.get("readout", 1))
    r3 = ea.run_masked(model, cfg["arm"], tk.TASKS[task]["dims"], d3, n, cfg.get("T_fixed"), m=cfg.get("readout", 1))
    assert np.array_equal(r2["correct_v2"], r3["correct_v2"]) and np.allclose(r2["bitacc"], r3["bitacc"])
print("evaluate_a2 tests ok")
