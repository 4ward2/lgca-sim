"""Stage A2 task generators: synthetic inpainting, cartoon middle frame, Game of Life.

Same conventions as sim/tasks.py, plus a per-cell scoring mask:
  x    : (B, c, H, W) float32 binary input channels
  y    : (B, H, W)    float32 per-cell binary labels
  mask : (B, H, W)    bool, the cells that are scored
  diff : (B,)         int difficulty D
Ternary values {-1, 0, +1} (0 = unknown / erased) enter as two thermometer bits [v >= 0, v >= +1].
"""
import numpy as np


def encode_ternary(t):
    """(..., H, W) in {-1, 0, 1} -> (..., 2, H, W) thermometer bits."""
    t = np.asarray(t)
    return np.stack([t >= 0, t >= 1], axis=-3).astype(np.float32)


def decode_ternary(e):
    e = np.asarray(e)
    lo, hi = e[..., 0, :, :] > 0.5, e[..., 1, :, :] > 0.5
    return np.where(hi, 1, np.where(lo, 0, -1)).astype(np.int8)


TASKS_A2 = {
    # inpaint: n and hole side d; extrapolation grows both
    "inpaint": dict(dims=2, c=2, train_n=(12, 16), train_d=(2, 4), indist=(16, 4),
                    extra=((24, 8), (32, 16), (48, 24))),
    # midframe: n and max speed s (cells per frame, Chebyshev)
    "midframe": dict(dims=2, c=4, train_n=(12, 16), train_s=(1, 2), indist=(16, 2),
                     extra=((24, 4), (32, 8), (48, 16))),
    "life": dict(dims=2, c=1, train_n=(8, 16), indist=(16,), extra=((32,), (64,))),
    # Q: greatest-fixed-point control, sizes as for M
    "cycles": dict(dims=2, c=2, train_n=(6, 12), indist=(12,), extra=((16,), (24,), (32,))),
    # Q+: least-fixed-point positive control on the same graphs (+1 source bit)
    "cycles_reach": dict(dims=2, c=3, train_n=(6, 12), indist=(12,), extra=((16,), (24,), (32,))),
}


# ---------- inpaint ----------

def _rects(rng, n):
    """1-4 axis-aligned rectangles (x0, y0, x1, y1 exclusive) painted in order."""
    out = []
    for _ in range(int(rng.integers(1, 5))):
        x0, y0 = int(rng.integers(0, n - 1)), int(rng.integers(0, n - 1))
        x1, y1 = int(rng.integers(x0 + 1, n + 1)), int(rng.integers(y0 + 1, n + 1))
        out.append((x0, y0, x1, y1, int(rng.choice([-1, 1]))))
    return out


def _corner_in(rect, hx, hy, d):
    """True if any of the rectangle's 4 corner cells lies inside the d x d hole at (hx, hy)."""
    x0, y0, x1, y1, _ = rect
    for cx in (x0, x1 - 1):
        for cy in (y0, y1 - 1):
            if hx <= cx < hx + d and hy <= cy < hy + d:
                return True
    return False


def gen_inpaint(rng, B, n, d=None, dmax=4, unambiguous=True, edge_frac=0.5):
    """Background +-1 with 1-4 painted rectangles; one d x d hole erased to 0.
    unambiguous=True rejects holes that contain a rectangle corner: the hole content is then
    fixed by extending the rectangle edges that cross the hole border (a unique answer).
    edge_frac: each sample is, with this probability, required to have an edge crossing the hole
    (non-constant hole content); otherwise unconstrained (mostly single-colour holes). Returned as
    `edge` (bool per sample) for stratified reporting (brief_A2 §5)."""
    xs, ys, ms, ds, es = [], [], [], [], []
    want_edge = None
    while len(xs) < B:
        if want_edge is None:  # drawn once per output sample, kept across rejections
            want_edge = rng.random() < edge_frac
        clean = np.full((n, n), int(rng.choice([-1, 1])), np.int8)
        rects = _rects(rng, n)
        for x0, y0, x1, y1, v in rects:
            clean[y0:y1, x0:x1] = v
        dd = int(d if d is not None else rng.integers(2, dmax + 1))
        if dd > n - 2:
            raise ValueError("hole must leave a known border")
        hx, hy = int(rng.integers(1, n - dd)), int(rng.integers(1, n - dd))  # hole strictly inside
        if unambiguous and any(_corner_in(r, hx, hy, dd) for r in rects):
            continue
        mask = np.zeros((n, n), bool)
        mask[hy:hy + dd, hx:hx + dd] = True
        is_edge = np.unique(clean[mask]).size > 1
        if want_edge and not is_edge:
            continue
        erased = clean.copy()
        erased[mask] = 0
        xs.append(encode_ternary(erased))
        ys.append((clean == 1).astype(np.float32))
        ms.append(mask)
        ds.append((dd + 1) // 2)
        es.append(is_edge)
        want_edge = None
    return dict(x=np.stack(xs), y=np.stack(ys), mask=np.stack(ms), diff=np.array(ds, np.int64),
                edge=np.array(es, bool))


# ---------- midframe ----------

SHAPES = {
    "plus": [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
}


def _sprite_cells(kind, w, h):
    if kind == "plus":
        return SHAPES["plus"]
    return [(dx, dy) for dy in range(h) for dx in range(w)]


def _render(sprites, n, t):
    """Binary frame at time t; sprites are (kind, w, h, x, y, vx, vy)."""
    a = np.zeros((n, n), np.int8)
    for kind, w, h, x, y, vx, vy in sprites:
        for dx, dy in _sprite_cells(kind, w, h):
            a[y + vy * t + dy, x + vx * t + dx] = 1
    return a


def _cells(sp, t):
    kind, w, h, x, y, vx, vy = sp
    return {(x + vx * t + dx, y + vy * t + dy) for dx, dy in _sprite_cells(kind, w, h)}


def gen_midframe(rng, B, n, smax=None, s_lo=1):
    """1-3 binary sprites (filled rectangles 2-4 or a plus) moving linearly with integer velocity,
    |v|_inf in [s_lo, smax] for the fastest sprite; frames t=0,1,2 stay inside the canvas and sprites
    never overlap (including their t=0..2 footprints at the same time step).
    Input: f0 and f2 as thermometer bits of {-1 (empty), +1 (object)}; target: f1."""
    smax = 2 if smax is None else smax
    xs, ys, ms, ds = [], [], [], []
    while len(xs) < B:
        sprites = []
        for _ in range(int(rng.integers(1, 4))):
            kind = "plus" if rng.integers(2) else "rect"
            w = h = 3 if kind == "plus" else 0
            if kind == "rect":
                w, h = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            vx, vy = int(rng.integers(-smax, smax + 1)), int(rng.integers(-smax, smax + 1))
            # start position such that t=0..2 stay inside: x + 2*vx in [0, n-w]
            lo_x, hi_x = max(0, -2 * vx), min(n - w, n - w - 2 * vx)
            lo_y, hi_y = max(0, -2 * vy), min(n - h, n - h - 2 * vy)
            if lo_x > hi_x or lo_y > hi_y:
                sprites = None
                break
            sprites.append((kind, w, h, int(rng.integers(lo_x, hi_x + 1)), int(rng.integers(lo_y, hi_y + 1)), vx, vy))
        if not sprites:
            continue
        speed = max(max(abs(s[5]), abs(s[6])) for s in sprites)
        if speed < s_lo:
            continue
        overlap = any(_cells(a, t) & _cells(b, t) for t in range(3)
                      for i, a in enumerate(sprites) for b in sprites[i + 1:])
        if overlap:
            continue
        f0, f1, f2 = (_render(sprites, n, t) for t in range(3))
        tern = lambda f: np.where(f > 0, 1, -1)
        xs.append(np.concatenate([encode_ternary(tern(f0)), encode_ternary(tern(f2))], 0))
        ys.append(f1.astype(np.float32))
        ms.append(np.ones((n, n), bool))
        ds.append(speed)
    return dict(x=np.stack(xs), y=np.stack(ys), mask=np.stack(ms), diff=np.array(ds, np.int64))


# ---------- Game of Life ----------

def life_step(a):
    """B3/S23 with dead cells outside the canvas. a: (..., H, W) bool."""
    H, W = a.shape[-2:]
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)])
    nb = sum(p[..., i:i + H, j:j + W].astype(np.int8) for i in range(3) for j in range(3)) - a
    return (a & ((nb == 2) | (nb == 3))) | (~a & (nb == 3))


def gen_life(rng, B, n, density=0.3):
    a = rng.random((B, n, n)) < density
    return dict(x=a[:, None].astype(np.float32), y=life_step(a).astype(np.float32),
                mask=np.ones((B, n, n), bool), diff=np.ones(B, np.int64))


def generate(task, rng, B, n, **kw):
    return {"inpaint": gen_inpaint, "midframe": gen_midframe, "life": gen_life,
            "cycles": gen_cycles, "cycles_reach": gen_cycles_reach}[task](rng, B, n, **kw)


def ascii(frame):
    return "\n".join("".join({-1: ".", 0: "?", 1: "#"}.get(int(v), str(int(v))) for v in row) for row in frame)


# ---------- Q: cycle membership on an arrow grid (greatest-fixed-point control) ----------

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # 0=N 1=E 2=S 3=W as (dx, dy)


def _rect_loop(x0, y0, w, h, cw):
    """Perimeter cells of a w x h rectangle (w, h >= 2) in walking order."""
    p = [(x0 + i, y0) for i in range(w)] + [(x0 + w - 1, y0 + j) for j in range(1, h)]
    p += [(x0 + i, y0 + h - 1) for i in range(w - 2, -1, -1)] + [(x0, y0 + j) for j in range(h - 2, 0, -1)]
    return p if cw else p[::-1]


def _dir_to(a, b):
    return DIRS.index((b[0] - a[0], b[1] - a[1]))


def gen_cycles(rng, B, n, max_loops=3, min_loops=2):
    """Each cell holds an arrow (2 bits) to an in-bounds 4-neighbour. 2-3 planted rectangle-perimeter
    cycles; every other cell points to a random BFS parent toward the cycles, so the planted loops are
    the only cycles. Label = cell lies on a cycle. D = number of leaf-peeling rounds (longest tail)."""
    xs, ys, ms, ds = [], [], [], []
    while len(xs) < B:
        occ = np.zeros((n, n), bool)
        dirs = np.full((n, n), -1, np.int64)
        loops = 0
        for _ in range(int(rng.integers(1, max_loops + 1))):
            w, h = int(rng.integers(2, max(3, n // 2) + 1)), int(rng.integers(2, max(3, n // 2) + 1))
            if w > n or h > n:
                continue
            x0, y0 = int(rng.integers(0, n - w + 1)), int(rng.integers(0, n - h + 1))
            if occ[max(0, y0 - 1):y0 + h + 1, max(0, x0 - 1):x0 + w + 1].any():  # keep loops apart
                continue
            p = _rect_loop(x0, y0, w, h, bool(rng.integers(2)))
            for k, c in enumerate(p):
                dirs[c[1], c[0]] = _dir_to(c, p[(k + 1) % len(p)])
            for c in p:
                occ[c[1], c[0]] = True
            loops += 1
        if loops < min_loops:
            continue
        on = occ.copy()
        # BFS outward from cycle cells; each new cell points to a random already-assigned neighbour
        frontier = [(x, y) for y in range(n) for x in range(n) if on[y, x]]
        assigned = on.copy()
        while frontier:
            nxt = []
            order = rng.permutation(len(frontier))
            cand = {}
            for i in order:
                fx, fy = frontier[i]
                for dx, dy in DIRS:
                    x, y = fx + dx, fy + dy
                    if 0 <= x < n and 0 <= y < n and not assigned[y, x]:
                        cand.setdefault((x, y), []).append((fx, fy))
            for (x, y), parents in cand.items():
                px, py = parents[int(rng.integers(len(parents)))]
                dirs[y, x] = _dir_to((x, y), (px, py))
                assigned[y, x] = True
                nxt.append((x, y))
            frontier = nxt
        lab, rounds = cycle_labels(dirs)
        x = np.stack([(dirs >> 1) & 1, dirs & 1], 0).astype(np.float32)
        xs.append(x)
        ys.append(lab.astype(np.float32))
        ms.append(np.ones((n, n), bool))
        ds.append(rounds)
    return dict(x=np.stack(xs), y=np.stack(ys), mask=np.stack(ms), diff=np.array(ds, np.int64))


def cycle_labels(dirs):
    """Peel cells with in-degree 0 until nothing changes (greatest fixed point). Returns (on_cycle, rounds)."""
    n = dirs.shape[0]
    alive = np.ones_like(dirs, bool)
    tx = np.zeros_like(dirs); ty = np.zeros_like(dirs)
    for k, (dx, dy) in enumerate(DIRS):
        sel = dirs == k
        yy, xx = np.nonzero(sel)
        tx[sel], ty[sel] = xx + dx, yy + dy
    rounds = 0
    while True:
        indeg = np.zeros_like(dirs)
        yy, xx = np.nonzero(alive)
        np.add.at(indeg, (ty[yy, xx], tx[yy, xx]), 1)
        drop = alive & (indeg == 0)
        if not drop.any():
            return alive, rounds
        alive &= ~drop
        rounds += 1


def reach_source_labels(dirs, src):
    """Q+: cells whose pointer chain reaches the source cell src=(x, y) (least fixed point:
    grow backwards from the source along reversed arrows). Returns (label, longest path length)."""
    n = dirs.shape[0]
    tx = np.zeros_like(dirs); ty = np.zeros_like(dirs)
    for k, (dx, dy) in enumerate(DIRS):
        sel = dirs == k
        yy, xx = np.nonzero(sel)
        tx[sel], ty[sel] = xx + dx, yy + dy
    lab = np.zeros((n, n), bool)
    lab[src[1], src[0]] = True
    rounds = 0
    while True:
        new = ~lab & lab[ty, tx]
        if not new.any():
            return lab, rounds
        lab |= new
        rounds += 1


def gen_cycles_reach(rng, B, n, max_loops=3, min_loops=2):
    """Q+ (positive control for Q): same arrow grids as gen_cycles but with >= 2 loops (with a single
    loop every cell reaches the source and the all-ones answer is trivially correct); one random
    on-cycle cell is marked as the source (extra input bit). Label = the cell's pointer chain reaches it."""
    base = gen_cycles(rng, B, n, max_loops, min_loops)  # identical graphs to gen_cycles(rng, ...) (Q/Q+ pairing)
    src_rng = np.random.default_rng(int(rng.integers(2 ** 62)))  # separate stream for the source choice
    xs, ys, ds = [], [], []
    for k in range(B):
        dirs = (base["x"][k, 0] > 0.5).astype(np.int64) * 2 + (base["x"][k, 1] > 0.5).astype(np.int64)
        cyc = np.argwhere(base["y"][k] > 0.5)
        sy, sx = cyc[int(src_rng.integers(len(cyc)))]
        lab, rounds = reach_source_labels(dirs, (int(sx), int(sy)))
        srcbit = np.zeros((n, n), np.float32)
        srcbit[sy, sx] = 1
        xs.append(np.concatenate([base["x"][k], srcbit[None]], 0))
        ys.append(lab.astype(np.float32))
        ds.append(rounds)
    return dict(x=np.stack(xs), y=np.stack(ys), mask=np.ones((B, n, n), bool), diff=np.array(ds, np.int64))
