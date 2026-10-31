"""Minimal pure-Python DPLL (watched literals, chronological backtracking, no learning).  FOR TESTS ONLY: used to validate the
k2_sat encoder on tiny instances because no SAT solver is installed locally."""
import sys


class Dpll:
    def __init__(self, nv, clauses):
        self.nv = nv
        self.val = [0] * (nv + 1)  # 0 unassigned, +1 true, -1 false
        self.trail = []
        self.watch = {}
        self.clauses = []
        self.ok = True
        self.units = []
        for cl in clauses:
            cl = list(dict.fromkeys(cl))
            if any(-l in cl for l in cl):
                continue  # tautology
            if not cl:
                self.ok = False
            elif len(cl) == 1:
                self.units.append(cl[0])
            else:
                i = len(self.clauses)
                self.clauses.append(cl)
                self.watch.setdefault(cl[0], []).append(i)
                self.watch.setdefault(cl[1], []).append(i)
        self.qhead = 0

    def value(self, lit):
        v = self.val[abs(lit)]
        return v if lit > 0 else -v

    def assign(self, lit):
        v = self.value(lit)
        if v == 1:
            return True
        if v == -1:
            return False
        self.val[abs(lit)] = 1 if lit > 0 else -1
        self.trail.append(lit)
        return True

    def propagate(self):
        while self.qhead < len(self.trail):
            lit = self.trail[self.qhead]
            self.qhead += 1
            false_lit = -lit
            ws = self.watch.get(false_lit, [])
            keep = []
            conflict = False
            for k, ci in enumerate(ws):
                if conflict:
                    keep.append(ci)
                    continue
                cl = self.clauses[ci]
                if cl[0] == false_lit:
                    cl[0], cl[1] = cl[1], cl[0]
                if self.value(cl[0]) == 1:
                    keep.append(ci)
                    continue
                for j in range(2, len(cl)):
                    if self.value(cl[j]) != -1:
                        cl[1], cl[j] = cl[j], cl[1]
                        self.watch.setdefault(cl[1], []).append(ci)
                        break
                else:
                    keep.append(ci)
                    if self.value(cl[0]) == -1 or not self.assign(cl[0]):
                        conflict = True
            self.watch[false_lit] = keep
            if conflict:
                return False
        return True

    def backtrack_to(self, n):
        while len(self.trail) > n:
            self.val[abs(self.trail.pop())] = 0
        self.qhead = min(self.qhead, n)

    def solve(self, assumptions=(), decision_vars=()):
        """Returns model (list of true var ids) or None.  Decisions only on decision_vars first (then any var)."""
        sys.setrecursionlimit(100000)
        if not self.ok:
            return None
        for l in list(self.units) + list(assumptions):
            if not self.assign(l):
                return None
        if not self.propagate():
            return None
        order = list(decision_vars) + [v for v in range(1, self.nv + 1)]
        stack = []  # (trail length before decision, var, tried_both)
        pos = 0
        while True:
            while pos < len(order) and self.val[order[pos]] != 0:
                pos += 1
            if pos >= len(order):
                return [v for v in range(1, self.nv + 1) if self.val[v] == 1]
            v = order[pos]
            stack.append([len(self.trail), v, False, pos])
            self.assign(-v)  # default phase: false
            while not self.propagate():
                # backtrack
                while stack and stack[-1][2]:
                    stack.pop()
                if not stack:
                    return None
                top = stack[-1]
                self.backtrack_to(top[0])
                top[2] = True
                pos = top[3]
                self.assign(top[1])
