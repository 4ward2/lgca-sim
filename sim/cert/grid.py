"""Sufficient all-size reachability certificate (UNKNOWN is not failure).

Input contract from tasks.gen_maze: a finite nonempty rectangular grid, at least
one open cell, exactly one source, which is open. Codes: wall=4, open=5,
source=7, virtual=0. We prove the stronger multi-source contract as well.

Invariant: each open cell retains its type-specific true reachable value; walls
retain their unary domain W. Source unary anchors true. Walls reach W in one
round and remain there; no source-free component can lose either open value,
since full open domains with W/virtual neighbours are locally fixed. Induct on
distance from a source: a singleton neighbour forces the next open cell to its
true singleton. Safety prevents deletion of that singleton by other ports.
Readout checks finish both component cases and walls. No contradiction occurs.
The transient wall top is a superset of W, so monotonicity covers round zero.
"""
import itertools

from .tables import unary, support


def certificate(t):
    if t['V'] != 2 or t['dims'] != 2:
        return dict(verdict='UNKNOWN', violated_condition='requires V=2, dims=2')
    full = 3; wall = unary(t, 4)
    failures = []
    for flips in itertools.product((0, 1), repeat=3):
        pi = dict(zip((4, 5, 7), flips))
        true = {x: 1 << (1 ^ pi[x]) for x in (5, 7)}

        def check():
            if not wall:
                return 'wall unary is empty'
            if unary(t, 7) != true[7]:
                return 'source unary anchor'
            if unary(t, 5) != full:
                return 'source-free open unary must be full'
            if t['o'][4][wall] != 0 or t['o'][5][full] != 0:
                return 'wall/unknown readout'
            if any(t['o'][x][true[x]] != 1 for x in (5, 7)):
                return 'reachable singleton readout'
            for e, right in itertools.product(range(2), (True, False)):
                def incoming(x, y, d):
                    return (support(t, e, y, x, d) if right else
                            support(t, e, x, y, d, False))
                # Safety under all invariant neighbour domains (including source).
                for x in (4, 5, 7):
                    keep = wall if x == 4 else true[x]
                    for y, d in ((0, full), (4, wall), (5, true[5]), (7, true[7])):
                        if incoming(x, y, d) & keep != keep:
                            return f'true/wall support: port={e},{right}, types={x},{y}'
                # Source-free H must be fixed on every port; walls cannot anchor.
                for y, d in ((0, full), (4, wall), (5, full)):
                    if incoming(5, y, d) != full:
                        return f'source-free unknown fixed point: port={e},{right}, neighbour={y}'
                for y in (5, 7):
                    if incoming(5, y, true[y]) != true[5]:
                        return f'one known neighbour progress: port={e},{right}, neighbour={y}'
            return None

        last = check()
        if last is None:
            return dict(verdict='PASS', pi=pi, wall_domain=wall,
                        contract='rectangular grid; >=1 open; exactly one source on open (also valid for multiple sources)')
        failures.append(dict(pi=pi, violated_condition=last))
    return dict(verdict='UNKNOWN', violated_condition=last, candidates=8, candidate_failures=failures)


def decide(t, max_cells=4, time_limit=5):
    out = certificate(t)
    if out['verdict'] == 'UNKNOWN':
        from .grid_exhaustive import search
        result = search(t, max_cells, time_limit)
        out['bounded_search'] = result
        if result['verdict'] == 'FAIL':
            out.update(verdict='FAIL', counterexample=result['counterexample'],
                       simulator_prediction=result['simulator_prediction'])
    return out
