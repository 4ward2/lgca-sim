import unittest
from itertools import product
from . import oracle as o


def independent(puzzle, n):
    """Plain chronological backtracking, independent of candidate propagation."""
    regions = o.standard_regions(n)
    board = list(puzzle)
    answers = []
    def visit(c):
        if c == n*n:
            answers.append(tuple(board))
            return
        choices = range(n) if board[c] == -1 else [board[c]]
        for v in choices:
            if all(all(j == c or board[j] != v for j in r) for r in regions if c in r):
                old = board[c]
                board[c] = v
                visit(c+1)
                board[c] = old
    visit(0)
    return answers


class Oracles(unittest.TestCase):
    def test_counts(self):
        sol = tuple((r*2+r//2+c) % 4 for r in range(4) for c in range(4))
        self.assertTrue(o.validate_solution(sol, 4))
        for p in [sol, (-1,)*16, (0,0)+(-1,)*14, (-1,)+sol[1:]]:
            self.assertEqual(o.count_solutions(p,4), min(2,len(independent(p,4))))
        self.assertEqual(o.count_solutions((-1,)*16,4,limit=400),288)
        self.assertFalse(o.validate_solution((0,)*16,4))
        self.assertEqual(o.singles_closure((-1,)*16,4),[set(range(4)) for _ in range(16)])
        self.assertEqual(o.singles_closure((-1,)+sol[1:],4),[{v} for v in sol])
        nine = tuple((r*3+r//3+c)%9 for r in range(9) for c in range(9))
        p = (-1,)+nine[1:]
        self.assertEqual(o.count_solutions(p,9),len(independent(p,9)))
    def test_hall(self):
        d = [set(range(4)) for _ in range(16)]
        d[0]=d[1]={0,1}
        h=o.hall_closure((-1,)*16,4,candidates=d)
        self.assertEqual(h[2],{2,3})
        bad=[set(range(4)) for _ in range(16)]
        bad[0]=bad[1]={0}
        self.assertTrue(any(not x for x in o.singles_closure((-1,)*16,4,candidates=bad)))
        zero=[set(range(1,4)) for _ in range(16)]
        self.assertTrue(any(not x for x in o.singles_closure((-1,)*16,4,candidates=zero)))

if __name__ == '__main__':
    unittest.main()
