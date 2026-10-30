"""Sufficient all-size reachability certificate (UNKNOWN is not failure).

Input contract from tasks.gen_maze: a finite nonempty rectangular grid, at least
one open cell, exactly one source, which is open. Codes: wall=4, open=5,
source=7, virtual=0. We prove the stronger multi-source contract as well.

Invariant: each open cell retains its type-specific true reachable value; walls
retain a nonempty core W and stay within U(4) after round one. Source unary
anchors true. Wall domains remain between W and U(4) after round one;
no source-free component can lose either open value,
since full open domains with W/virtual neighbours are locally fixed. Induct on
distance from a source: a singleton neighbour forces the next open cell to its
true singleton. Safety prevents deletion of that singleton by other ports.
Readout checks finish both component cases and walls. No contradiction occurs.
The transient wall top is a superset of W, so monotonicity covers round zero.
"""
import itertools

from .tables import unary, support, validate, provenance


def _obligations(t, pi, wall):
    full = 3; wall_unary = unary(t, 4)
    true = {x: 1 << (1 ^ pi[x]) for x in (5, 7)}
    obligations = []

    def record(kind, actual, expected, relation='equal', **context):
        passed = (actual == expected if relation == 'equal' else
                  bool(actual) if relation == 'nonempty' else actual & expected == expected)
        obligations.append(dict(kind=kind, actual=actual, expected=expected,
                                relation=relation, passed=passed, **context))

    record('unary', wall_unary, wall, 'contains', x=4)
    record('unary', unary(t, 7), true[7], x=7)
    record('unary', unary(t, 5), full, x=5)
    wall_domains = [d for d in range(1, 4) if d & wall == wall and d & wall_unary == d]
    for x, d, y in ([(4, d, 0) for d in wall_domains] + [(5, full, 0), (5, true[5], 1), (7, true[7], 1)]):
        record('readout', t['o'][x][d], y, x=x, domain=d)
    for e, right in itertools.product(range(2), (True, False)):
        def incoming(x, y, d, expected, relation='equal'):
            actual = (support(t, e, y, x, d) if right else support(t, e, x, y, d, False))
            record('support', actual, expected, relation, axis=e, target_right=right,
                   target=x, neighbour=y, domain=d)
        for x in (4, 5, 7):
            for y, d in ((0, full), (4, wall), (5, true[5]), (7, true[7])):
                incoming(x, y, d, wall if x == 4 else true[x], 'contains')
        for y, d in ((0, full), (4, wall), (5, full)):
            incoming(5, y, d, full)
        for y in (5, 7):
            incoming(5, y, true[y], true[5])
    return obligations


def replay(t, obligations):
    """Check the complete recorded proof for its chosen core/relabelling, no search."""
    validate(t, 'M')
    if t['V'] != 2:
        return False
    try:
        core = obligations[0]['expected']
        if core not in (1, 2, 3) or core & unary(t, 4) != core:
            return False
        pi = {}
        for x in (5, 7):
            truth = [o['domain'] for o in obligations if o['kind'] == 'readout'
                     and o['x'] == x and o['expected'] == 1]
            if len(truth) != 1 or truth[0] not in (1, 2):
                return False
            pi[x] = 1 ^ (truth[0].bit_length() - 1)
        expected = _obligations(t, pi, core)
        return obligations == expected and all(o['passed'] for o in expected)
    except (KeyError, TypeError, IndexError):
        return False


def certificate(t):
    validate(t, 'M')
    if t['V'] != 2:
        return dict(verdict='UNKNOWN', violated_condition='requires V=2', **provenance())
    full = 3; wall_unary = unary(t, 4)
    failures = []
    # Wall is a domain W, not a selected truth value: pi[4] was redundant.
    wall_candidates = list(dict.fromkeys((wall_unary, 1, 2, 3)))
    wall_candidates = [w for w in wall_candidates if w and w & wall_unary == w]
    if not wall_candidates:
        return dict(verdict='UNKNOWN', violated_condition='empty wall unary', **provenance())
    for wall, flips in itertools.product(wall_candidates, itertools.product((0, 1), repeat=2)):
        pi = dict(zip((5, 7), flips))
        true = {x: 1 << (1 ^ pi[x]) for x in (5, 7)}
        obligations = _obligations(t, pi, wall)
        failed = [o for o in obligations if not o['passed']]
        if not failed:
            assert replay(t, obligations)
            return dict(verdict='PASS', pi=pi, wall_domain=wall, obligations=obligations,
                        contract='rectangular grid; >=1 open; one or more sources on open', **provenance())
        failures.append(dict(pi=pi, violated_condition=failed[0]))
    return dict(verdict='UNKNOWN', violated_condition=failures[-1]['violated_condition'],
                candidates=len(failures), candidate_failures=failures, **provenance())


def decide(t, max_cells=4, time_limit=5, multi_source=False, witness=None):
    out = certificate(t)
    if out['verdict'] == 'UNKNOWN' and witness is not None:
        import numpy as np
        from .tables import validate_input, confirm
        from .grid_exhaustive import bfs
        xi = validate_input(t, witness)
        if not np.isin(xi, (4, 5, 7)).all() or not (xi == 7).any():
            raise ValueError('witness must be a legal grid with a source')
        prediction = confirm(t, xi)
        expected = bfs(xi).tolist()
        if prediction != expected:
            out.update(verdict='FAIL', counterexample=xi.tolist(), expected=expected,
                       simulator_prediction=prediction)
    if out['verdict'] == 'UNKNOWN':
        from .grid_exhaustive import search
        result = search(t, max_cells, time_limit, multi_source=multi_source)
        out['bounded_search'] = result
        if result['verdict'] == 'FAIL':
            out.update(verdict='FAIL', counterexample=result['counterexample'],
                       simulator_prediction=result['simulator_prediction'])
    return out
