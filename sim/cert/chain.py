"""Complete finite-word decision procedure for P and A, including empty domains.

Virtual boundary cells are immutable full domains with input 0. Each boundary
edge is therefore exactly an existential unary restriction of its real endpoint.
After folding those restrictions, the constraint graph is a path. On a tree,
nonempty arc-consistent domains extend to a global assignment by choosing a root
value and recursively choosing supports. An empty domain propagates throughout
the connected path. Thus the closure equals the intersection of left-reachable
and right-reachable domains. No boundary unary u[0] is applied.

The finite left and right subset automata below enumerate all reachable states,
retaining shortest representative words. Combining at a shared letter is exact:
the two halves have no other shared variables. Length categories distinguish
singleton words from longer words, needed for addition's mandatory high zero.
Every pair is checked; the minimum failing combination is a shortest witness.
Synchronous convergence takes <= V*n strict rounds, plus a stability check.
"""
from collections import deque

from .tables import unary, support, interpret, confirm, validate, provenance


def decide(t, task):
    if task not in ('P', 'A'):
        raise ValueError('chain task must be P or A')
    validate(t, task)
    alphabet = (2, 3) if task == 'P' else (4, 5, 6, 7)
    full = (1 << t['V']) - 1

    def spec(x, prev):
        if task == 'P':
            y = prev ^ (x & 1)
            return y, y
        z = (x & 1) + ((x >> 1) & 1) + prev
        return z >> 1, z & 1

    # Left state: last letter, F, outgoing spec state, expected bit, length class.
    left = {}; queue = deque()
    for x in alphabet:
        st, y = spec(x, 0)
        k = (x, unary(t, x) & support(t, 0, 0, x, full), st, y, 1)
        left[k] = [x]; queue.append(k)
    while queue:
        k = queue.popleft(); x, f, st, _, _ = k
        for z in alphabet:
            ns, y = spec(z, st)
            nk = (z, unary(t, z) & support(t, 0, x, z, f), ns, y, 2)
            if nk not in left:
                left[nk] = left[k] + [z]; queue.append(nk)
    right = {}; queue = deque()
    for x in (alphabet if task == 'P' else (4,)):
        k = (x, unary(t, x) & support(t, 0, x, 0, full, False), 1)
        right[k] = [x]; queue.append(k)
    while queue:
        k = queue.popleft(); x, b, _ = k
        for z in alphabet:
            nk = (z, unary(t, z) & support(t, 0, z, x, b, False), 2)
            if nk not in right:
                right[nk] = [z] + right[k]; queue.append(nk)
    best = None
    for (x, f, _, y, ll), prefix in left.items():
        for (z, b, rl), suffix in right.items():
            if x != z or (task == 'A' and ll == rl == 1):
                continue
            if t['o'][x][f & b] != y:
                word = prefix + suffix[1:]
                if best is None or len(word) < len(best):
                    best = word
    out = dict(verdict='PASS' if best is None else 'FAIL',
               left_states=len(left), right_states=len(right), rounds_bound='V*n + 1 stability check')
    if best is not None:
        _, pred, _ = interpret(t, best)
        st = 0; expected = []
        for x in best:
            st, y = spec(x, st); expected.append(y)
        assert pred.tolist() != expected
        out.update(counterexample=best, expected=expected, simulator_prediction=confirm(t, best))
    out.update(provenance())
    return out
