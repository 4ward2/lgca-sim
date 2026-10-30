"""Extract, cross-check and certify all deployed runs without training."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import chain, grid
from .tables import load, confirm


def equivalence_key(t, task):
    """Canonical tables under per-type latent permutations, discarding illegal codes.

    Virtual u/o and virtual-virtual relations are never consulted and omitted.
    Full virtual domains permit arbitrary virtual latent permutation as well.
    This is a sufficient exact table equivalence, not behavioural minimization.
    """
    types = (2, 3) if task == 'P' else ((4, 5, 6, 7) if task == 'A' else (4, 5, 7))
    best = None
    for flips in itertools.product((0, 1), repeat=len(types) + 1):
        pi = dict(zip((0,) + types, flips)); data = []
        for x in types:
            data.extend(t['u'][x][h ^ pi[x]] for h in range(2))
            for s in range(4):
                old = sum(((s >> h) & 1) << (h ^ pi[x]) for h in range(2))
                data.append(t['o'][x][old])
        for e in range(t['dims']):
            for x, y in itertools.product((0,) + types, repeat=2):
                if x == y == 0:
                    continue
                data.extend(t['r'][e][x][y][h ^ pi[x]][j ^ pi[y]]
                            for h, j in itertools.product(range(2), repeat=2))
        key = tuple(data)
        if best is None or key < best:
            best = key
    return hashlib.sha256(bytes(best)).hexdigest()


def crosscheck(t, task, seed=50001):
    from sim import tasks
    rng = np.random.default_rng(seed)
    for n in (1, 2, 3, 5, 8, 13):
        d = tasks.generate(task, rng, 2, n)
        for x in d['x']:
            xi = sum(x[k].astype(int) << k for k in range(t['c'])) + (1 << t['c'])
            confirm(t, xi)
    if task == 'M':
        for h, w in ((1, 7), (7, 1), (2, 5), (5, 2)):
            xi = np.where(rng.random((h, w)) < .4, 4, 5)
            xi.flat[0] = 7
            confirm(t, xi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['P', 'A', 'M'], choices=['P', 'A', 'M'])
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []; classes = {}
    for task, seed in itertools.product(a.tasks, a.seeds):
        start = time.monotonic(); name = f'{task}_B_s{seed}'
        _, _, t = load(a.runs / name)
        crosscheck(t, task)
        result = chain.decide(t, task) if task != 'M' else grid.decide(t)
        key = task + ':' + equivalence_key(t, task)
        classes.setdefault(key, []).append(name)
        row = dict(run=name, task=task, seed=seed, table_hash=t['table_sha256'],
                   checkpoint_sha256=t['checkpoint_sha256'], seconds=time.monotonic()-start,
                   interpreter_crosscheck_seed=50001, **result)
        rows.append(row)
        (a.out / (name + '_tables.json')).write_text(json.dumps(t), encoding='utf-8')
        print(f"{name}: {row['verdict']} ({row['seconds']:.3f}s)", flush=True)
    report = dict(runs=rows, equivalence_classes=list(classes.values()),
                  scope='fixed-point readout; original evaluation halting window is not certified')
    (a.out / 'cert_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    md = '| Run | Verdict | Relabelling | Table SHA256 | Seconds |\n|---|---|---|---|---|\n'
    for r in rows:
        md += f"| {r['run']} | {r['verdict']} | {r.get('pi', 'n/a')} | {r['table_hash']} | {r['seconds']:.3f} |\n"
    (a.out / 'cert_report.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
