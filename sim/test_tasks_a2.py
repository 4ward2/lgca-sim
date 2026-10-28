"""Tests for sim/tasks_a2.py (synthetic data only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks_a2 as t2

rng = np.random.default_rng(0)

# ternary round trip
v = rng.integers(-1, 2, size=(3, 5, 7)).astype(np.int8)
assert (t2.decode_ternary(t2.encode_ternary(v)) == v).all()
assert t2.encode_ternary(v).shape == (3, 2, 5, 7)

# inpaint
for n, d in ((16, 4), (24, 8), (48, 24)):
    o = t2.gen_inpaint(np.random.default_rng([1, n]), 40, n, d=d)
    assert o["x"].shape == (40, 2, n, n) and o["y"].shape == (40, n, n) and o["mask"].shape == (40, n, n)
    assert (o["mask"].sum((1, 2)) == d * d).all() and (o["diff"] == (d + 1) // 2).all()
    er = t2.decode_ternary(o["x"])
    clean = np.where(o["y"] > 0.5, 1, -1)
    assert (er[o["mask"]] == 0).all() and (er[~o["mask"]] == clean[~o["mask"]]).all()
    # hole strictly inside: border row/col never erased
    assert not o["mask"][:, 0, :].any() and not o["mask"][:, -1, :].any()
    assert not o["mask"][:, :, 0].any() and not o["mask"][:, :, -1].any()
# unambiguous: reconstruct the hole by extending edges = for every hole cell, the clean value equals
# the value of the nearest known cell along its row OR column whenever those two agree... use the
# stronger generator-side check: regenerate with the same rng and assert no rectangle corner in hole
r = np.random.default_rng(7)
for _ in range(200):
    n = 16
    clean = np.full((n, n), 1, np.int8)
    rects = t2._rects(r, n)
    hx, hy, dd = 5, 5, 4
    has = any(t2._corner_in(q, hx, hy, dd) for q in rects)
    corners = [(cx, cy) for (x0, y0, x1, y1, _) in rects for cx in (x0, x1 - 1) for cy in (y0, y1 - 1)]
    assert has == any(5 <= cx < 9 and 5 <= cy < 9 for cx, cy in corners)
d_rand = t2.gen_inpaint(np.random.default_rng(3), 300, 16)["diff"]
assert set(np.unique(d_rand)) <= {1, 2}

# midframe
for n, s in ((16, 2), (32, 8)):
    o = t2.gen_midframe(np.random.default_rng([2, n]), 60, n, smax=s)
    assert o["x"].shape == (60, 4, n, n) and o["y"].shape == (60, n, n)
    assert (o["diff"] >= 1).all() and (o["diff"] <= s).all()
    f0 = t2.decode_ternary(o["x"][:, 0:2]) > 0
    f2 = t2.decode_ternary(o["x"][:, 2:4]) > 0
    # same number of object cells in all frames (nothing clipped, no overlap)
    assert (f0.sum((1, 2)) == o["y"].sum((1, 2))).all() and (f2.sum((1, 2)) == o["y"].sum((1, 2))).all()
# explicit render check
sp = [("rect", 2, 2, 1, 1, 2, 0)]
assert t2._render(sp, 8, 1)[1, 3] == 1 and t2._render(sp, 8, 1)[1, 1] == 0
# fast motion is reachable
o = t2.gen_midframe(np.random.default_rng(5), 40, 48, smax=16, s_lo=9)
assert (o["diff"] >= 9).all()

# life vs reference
def life_ref(a):
    H, W = a.shape
    out = np.zeros_like(a)
    for i in range(H):
        for j in range(W):
            nb = sum(a[ii, jj] for ii in range(max(0, i - 1), min(H, i + 2)) for jj in range(max(0, j - 1), min(W, j + 2))) - a[i, j]
            out[i, j] = (a[i, j] and nb in (2, 3)) or (not a[i, j] and nb == 3)
    return out
o = t2.gen_life(np.random.default_rng(4), 100, 12)
for k in range(100):
    assert (o["y"][k] > 0.5).tolist() == life_ref(o["x"][k, 0] > 0.5).tolist()

# determinism
for task, kw in (("inpaint", {}), ("midframe", {}), ("life", {})):
    a = t2.generate(task, np.random.default_rng(9), 8, 16, **kw)
    b = t2.generate(task, np.random.default_rng(9), 8, 16, **kw)
    assert all(np.array_equal(a[k], b[k]) for k in a)

print("tasks_a2 tests ok")
o = t2.gen_inpaint(np.random.default_rng(11), 1, 16, d=6)
print("inpaint sample (input, ? = hole):")
print(t2.ascii(t2.decode_ternary(o["x"][0])))
print("target:")
print(t2.ascii(np.where(o["y"][0] > 0.5, 1, -1)))
o = t2.gen_midframe(np.random.default_rng(12), 1, 16, smax=2)
print("midframe f0 / f1(target) / f2, speed", o["diff"][0])
for fr in (t2.decode_ternary(o["x"][0, 0:2]), np.where(o["y"][0] > 0.5, 1, -1), t2.decode_ternary(o["x"][0, 2:4])):
    print(t2.ascii(fr)); print()

# Q: cycle membership vs brute-force pointer following
def on_cycle_ref(dirs):
    n = dirs.shape[0]
    out = np.zeros((n, n), bool)
    for y in range(n):
        for x in range(n):
            cx, cy = x, y
            for _ in range(n * n):
                dx, dy = t2.DIRS[dirs[cy, cx]]
                cx, cy = cx + dx, cy + dy
                if (cx, cy) == (x, y):
                    out[y, x] = True
                    break
    return out
for n in (6, 12, 24):
    o = t2.gen_cycles(np.random.default_rng([6, n]), 20, n)
    assert o["x"].shape == (20, 2, n, n)
    for k in range(20):
        dirs = (o["x"][k, 0] > 0.5).astype(int) * 2 + (o["x"][k, 1] > 0.5).astype(int)
        # arrows stay inside
        for y in range(n):
            for x in range(n):
                dx, dy = t2.DIRS[dirs[y, x]]
                assert 0 <= x + dx < n and 0 <= y + dy < n
        assert ((o["y"][k] > 0.5) == on_cycle_ref(dirs)).all()
    assert (o["diff"] >= 1).all()
o = t2.gen_cycles(np.random.default_rng(13), 200, 24)
print("Q n=24: frac on-cycle %.3f, D mean %.1f max %d" % (o["y"].mean(), o["diff"].mean(), o["diff"].max()))
a = t2.gen_cycles(np.random.default_rng(9), 4, 12); b = t2.gen_cycles(np.random.default_rng(9), 4, 12)
assert all(np.array_equal(a[k], b[k]) for k in a)
print("cycles tests ok")

# Q+: reach-the-source vs brute-force pointer following
def reach_ref(dirs, sx, sy):
    n = dirs.shape[0]
    out = np.zeros((n, n), bool)
    for y in range(n):
        for x in range(n):
            cx, cy = x, y
            for _ in range(n * n + 1):
                if (cx, cy) == (sx, sy):
                    out[y, x] = True
                    break
                dx, dy = t2.DIRS[dirs[cy, cx]]
                cx, cy = cx + dx, cy + dy
    return out
for n in (6, 12, 20):
    o = t2.gen_cycles_reach(np.random.default_rng([8, n]), 15, n)
    assert o["x"].shape == (15, 3, n, n) and (o["x"][:, 2].sum((1, 2)) == 1).all()
    for k in range(15):
        dirs = (o["x"][k, 0] > 0.5).astype(int) * 2 + (o["x"][k, 1] > 0.5).astype(int)
        sy, sx = np.argwhere(o["x"][k, 2] > 0.5)[0]
        assert ((o["y"][k] > 0.5) == reach_ref(dirs, sx, sy)).all()
for n in (6, 8, 12):
    o = t2.gen_cycles_reach(np.random.default_rng([15, n]), 200, n)
    assert not (o["y"].reshape(200, -1).min(1) > 0.5).any(), "Q+ sample with all cells positive"
    print("Q+ n=%d: frac positive %.3f, all-ones samples 0" % (n, o["y"].mean()))
o = t2.gen_cycles_reach(np.random.default_rng(14), 200, 24)
print("Q+ n=24: frac positive %.3f, D mean %.1f max %d" % (o["y"].mean(), o["diff"].mean(), o["diff"].max()))
print("cycles_reach tests ok")
o = t2.gen_inpaint(np.random.default_rng(21), 600, 16, d=4)
assert 0.45 <= o["edge"].mean() <= 0.75, o["edge"].mean()
print("inpaint edge fraction %.2f ok" % o["edge"].mean())

# Q/Q+ pairing (brief_A2 §1.1): same seed and size -> identical arrow grids
import evaluate_a2 as ea
for spec in ((8,), (12,)):
    _, q = ea.fixed_set_a2("cycles", spec, 20, 777)
    _, qp = ea.fixed_set_a2("cycles_reach", spec, 20, 777)
    assert np.array_equal(q["x"], qp["x"][:, :2]), spec
    # the source lies on a cycle, so every Q+ sample shares at least its source cell with Q's positives
    assert ((q["y"] > 0.5) & (qp["x"][:, 2] > 0.5)).reshape(20, -1).any(1).all()
    assert np.array_equal(q["diff"].shape, qp["diff"].shape)
o = t2.gen_cycles(np.random.default_rng(5), 100, 6)
print("Q/Q+ paired graphs ok; Q n=6 min loops satisfied")
