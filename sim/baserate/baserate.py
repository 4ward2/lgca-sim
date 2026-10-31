"""Anti-false-positive base-rate control: do random deployed-shape tables ever certify PASS?

  python sim/baserate/baserate.py --task P|A|M --mode uniform|density --count N --seed S --out DIR --time-limit SEC
  python sim/baserate/baserate.py --task P --zeroprop --seed S --out DIR      # zero-propagation control
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sim.cert import chain, grid, tables

FIXTURES = ROOT / 'sim' / 'cert' / 'fixtures'
SHAPE = {'P': (1, 1), 'A': (1, 2), 'M': (2, 2)}  # dims, c
INDIST = {'P': 16, 'A': 16, 'M': 12}

_PROV = None


def _cached_provenance():
    """Same content as tables.provenance(); computed once (it re-hashes the sources on every call)."""
    global _PROV
    if _PROV is None:
        _PROV = tables.provenance()
    return dict(_PROV)


# Speed only: the checkers' per-verdict provenance block is identical for every table in a run.
chain.provenance = _cached_provenance
grid.provenance = _cached_provenance


def real_tables(task):
    return [json.loads((FIXTURES / f'{task}_B_s{s}_tables.json').read_text()) for s in range(1, 11)]


def sample_uniform(task, rng, V=2):
    dims, c = SHAPE[task]
    nin = 1 << (c + 1)
    t = dict(dims=dims, c=c, V=V,
             u=rng.integers(0, 2, size=(nin, V)),
             r=rng.integers(0, 2, size=(dims, nin, nin, V, V)),
             o=rng.integers(0, 2, size=(nin, 1 << V)))
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in t.items()}


# trailing array dims that form one "slice": u row (V bits), r[e][a][b] block (V*V bits), o row (2^V bits)
_SLICE_DIMS = {'u': 1, 'r': 2, 'o': 1}


def sample_density(task, rng, reals):
    """Every slice (u row, r[e][a][b] VxV block, o row) is a random permutation of the bits of the same slice
    of a randomly chosen (independently per slice) real deployed table of this task, so the per-slice
    density (number of ones) is exactly that of a real table."""
    out = {}
    for key in ('u', 'r', 'o'):
        stack = np.stack([np.asarray(t[key], dtype=np.int8) for t in reals])  # (T, ...)
        sd = _SLICE_DIMS[key]
        flat = stack.reshape(len(reals), -1, int(np.prod(stack.shape[stack.ndim - sd:])))
        res = np.empty(flat.shape[1:], dtype=np.int8)
        pick = rng.integers(0, len(reals), size=flat.shape[1])
        for i in range(flat.shape[1]):
            res[i] = rng.permutation(flat[pick[i], i])
        out[key] = res.reshape(stack.shape[1:]).tolist()
    ref = reals[0]
    return dict(dims=ref['dims'], c=ref['c'], V=ref['V'], **out)


def classify(t, task, max_cells=9, bounded_time=60):
    """P/A: chain's complete decision (PASS/FAIL). M: grid.decide; if the sufficient certificate does not
    prove PASS the category is UNKNOWN, with the bounded exhaustive result (FAIL / BOUNDED_PASS / UNKNOWN=time)
    recorded as detail. (grid.decide itself would relabel bounded FAIL as FAIL; we keep it separate.)"""
    if task in ('P', 'A'):
        r = chain.decide(t, task)
        return r['verdict'], dict(cx=r.get('counterexample'))
    r = grid.decide(t, max_cells=max_cells, time_limit=bounded_time)
    if 'bounded_search' not in r:  # certificate decided it
        return r['verdict'], {}
    return 'UNKNOWN', dict(bounded=r['bounded_search']['verdict'], cx=r.get('counterexample'))


def wilson_upper(k, n, z=1.959964):
    """Upper end of the Wilson 95% (two-sided) interval for k successes in n trials."""
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + half) / d)


def run(task, mode, count, seed, out, time_limit):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reals = real_tables(task)
    counts = Counter(); passes = []; unknown_survivors = []; t0 = time.monotonic()
    for _ in range(count):
        t = sample_uniform(task, rng) if mode == 'uniform' else sample_density(task, rng, reals)
        tables.validate(t, task)
        cat, det = classify(t, task)
        counts[cat] += 1
        if cat == 'UNKNOWN':
            counts['UNKNOWN/bounded_' + det['bounded']] += 1
            if det['bounded'] != 'FAIL':
                unknown_survivors.append(t)
        if cat == 'PASS':
            passes.append(t)
        if time.monotonic() - t0 >= time_limit:
            break
    tested = counts['PASS'] + counts['FAIL'] + counts['UNKNOWN']
    k = counts['PASS']
    res = dict(task=task, mode=mode, seed=seed, requested=count, tested=tested,
               stopped_by_time_limit=tested < count, seconds=time.monotonic() - t0,
               counts=dict(counts), pass_count=k,
               pass_rate_wilson95_upper=wilson_upper(k, tested),
               pass_rate_rule_of_three_upper=(3 / tested if k == 0 and tested else None),
               statement=f'{k} of {tested} PASS, Wilson 95% upper bound {wilson_upper(k, tested):.2e}',
               pass_tables=passes, m_unknown_bounded_survivors=unknown_survivors)
    (out / 'result.json').write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------- zero-propagation control
def zero_prop(task, seed, n_samples=500):
    import torch
    from sim import tasks
    from sim.b_arm import TernaryNet
    rows = []
    for s in range(1, 11):
        t = json.loads((FIXTURES / f'{task}_B_s{s}_tables.json').read_text())
        m = TernaryNet(t['dims'], t['c'], t['V'])
        for k in ('u', 'r', 'o'):
            getattr(m, k).data.copy_(torch.tensor(t[k]))
        rng = np.random.default_rng([seed, s])
        d = tasks.generate(task, rng, n_samples, INDIST[task])
        x = d['x'].astype(int); y = d['y'].astype(int)
        xi = torch.tensor(sum(x[:, k] << k for k in range(t['c'])) + (1 << t['c']))
        c0 = torch.ones(tuple(xi.shape) + (t['V'],))
        c1 = m.update(c0, xi)
        cf = c0
        for _ in range(t['V'] * int(np.prod(xi.shape[1:])) + 1):
            nx = m.update(cf, xi)
            if torch.equal(nx, cf):
                break
            cf = nx
        row = dict(table=f'{task}_B_s{s}', n=n_samples)
        for name, c in (('rounds0', c0), ('rounds1', c1), ('fixed_point', cf)):
            pred = m.readout(c, xi).numpy().astype(int)
            ax = tuple(range(1, pred.ndim))
            row[name] = dict(exact_match=float((pred == y).all(axis=ax).mean()),
                             cell_accuracy=float((pred == y).mean()))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices='PAM', required=True)
    p.add_argument('--mode', choices=('uniform', 'density'), default='uniform')
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--time-limit', type=float, default=1e9, dest='time_limit')
    p.add_argument('--zeroprop', action='store_true', help='only the zero-propagation control on the 10 real tables')
    a = p.parse_args()
    if a.seed in (12345, 24680) or a.seed < 800000:
        raise SystemExit('seed must be >= 800000 and never 12345/24680')
    if a.zeroprop:
        rows = zero_prop(a.task, a.seed)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, f'zeroprop_{a.task}.json').write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(r['table'], {k: round(v['exact_match'], 3) for k, v in r.items() if isinstance(v, dict)})
        return
    res = run(a.task, a.mode, a.count, a.seed, a.out, a.time_limit)
    print(res['statement'], dict(res['counts']), f"{res['seconds']:.1f}s")


if __name__ == '__main__':
    main()
