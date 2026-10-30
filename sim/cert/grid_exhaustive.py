"""Budgeted exhaustive legal rectangular grids, including skinny rectangles."""
import argparse
import json
import time

import numpy as np

from .tables import load, interpret, confirm


def bfs(xi):
    reach = xi == 7
    while True:
        nxt = reach.copy()
        nxt[1:] |= reach[:-1]; nxt[:-1] |= reach[1:]
        nxt[:, 1:] |= reach[:, :-1]; nxt[:, :-1] |= reach[:, 1:]
        nxt &= xi != 4
        if np.array_equal(nxt, reach):
            return reach.astype(int)
        reach = nxt


def search(t, max_cells=9, time_limit=60):
    if max_cells < 1 or time_limit <= 0:
        raise ValueError('positive budgets required')
    start = time.monotonic(); count = 0; completed = []
    for cells in range(1, max_cells + 1):
        for h in range(1, cells + 1):
            if cells % h:
                continue
            w = cells // h
            for mask in range(1, 1 << cells):
                for source in range(cells):
                    if not mask & (1 << source):
                        continue
                    if time.monotonic() - start >= time_limit:
                        return dict(verdict='UNKNOWN', reason='time budget', checked=count, completed_shapes=completed)
                    xi = np.array([5 if mask & (1 << i) else 4 for i in range(cells)])
                    xi[source] = 7; xi = xi.reshape(h, w)
                    _, pred, _ = interpret(t, xi); expected = bfs(xi); count += 1
                    if not np.array_equal(pred, expected):
                        return dict(verdict='FAIL', checked=count, counterexample=xi.tolist(),
                                    expected=expected.tolist(), simulator_prediction=confirm(t, xi))
            completed.append([h, w])
    return dict(verdict='BOUNDED_PASS', checked=count, completed_shapes=completed,
                max_cells=max_cells, seconds=time.monotonic()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir'); p.add_argument('--max-cells', type=int, required=True)
    p.add_argument('--time-limit', type=float, required=True)
    a = p.parse_args()
    _, _, t = load(a.run_dir)
    print(json.dumps(search(t, a.max_cells, a.time_limit), indent=2))


if __name__ == '__main__':
    main()
