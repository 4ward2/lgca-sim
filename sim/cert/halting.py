"""Membership convergence, separate from correctness or early readout stability.

Let H^0 be top. The local support map is monotone and T(top) <= top;
induction gives H^{k+1} <= H^k. Every strict round deletes at least one
of V*cells bits. Consequently H^{V*cells} is already fixed (the extra
comparison round detects this but is not an additional propagation round).
For chains with V=2 this proves 2n <= 4n-ceil(4n/2), including n=1.
This bound is stronger in generality than a path-specific propagation proof:
it covers contradictory domains and all subset automata states as well.
For square M grids the bound 2n^2 exceeds floor(n^2/2); no halting claim.
PASS certifies convergence by the window deadline; combine with a separate
fixed-point correctness PASS to claim halted and correct at the horizon.
For P/A any first plateau detection occurs at time >= MH=2n, hence
its captured output is already the fixed-point output. Earlier plateau starts
cannot produce an incorrect captured output. This also covers A2 early capture.
"""
from .tables import validate, provenance


def decide(t, task):
    validate(t, task)
    chain = task in ('P', 'A')
    return dict(verdict='PASS' if chain and t['V'] <= 2 else 'UNKNOWN',
                fixed_point_bound=f"{t['V']}*" + ('n' if chain else 'cells'),
                window_deadline='2*n' if chain else 'floor(n*n/2)',
                proof='monotone iteration from top; each strict round deletes a membership bit',
                scope='P/A: final window and first plateau capture contain the fixed-point output; M: unproved',
                **provenance())
