import unittest
from .search import solve,undo_trail
from .oracle import count_solutions,validate_solution

class SearchTests(unittest.TestCase):
    def test_complete_and_budget(self):
        for p in ([-1]*16,[0]+[-1]*15,[0,1,-1,-1,-1,-1,0,-1,-1,0,-1,-1,-1,-1,-1,0]):
            r=solve(p,4,trace=True)
            self.assertEqual(r['status']=='SAT',count_solutions(p,4)>0)
            if r['solution']:
                self.assertTrue(validate_solution(r['solution'],4,p))
            self.assertEqual(r,dict(solve(p,4,trace=True),seconds=r['seconds'],propagation_seconds=r['propagation_seconds'],ordering_seconds=r['ordering_seconds']))
        self.assertEqual(solve([-1]*16,4,node_limit=0)['status'],'BUDGET')
        self.assertEqual(solve([-1]*16,4,time_limit=0)['status'],'BUDGET')
    def test_trail(self):
        d=[1,2]; t=[(0,3),(1,3)]; undo_trail(d,t,0)
        self.assertEqual(d,[3,3]); self.assertEqual(t,[])

    def test_same_order_and_random_oracle(self):
        import random
        from unittest.mock import patch
        from . import search
        rng=random.Random(60080)
        solution=[0,1,2,3,2,3,0,1,1,0,3,2,3,2,1,0]
        for _ in range(12):
            p=[v if rng.random()<.45 else -1 for v in solution]
            a=solve(p,4,trace=True)
            self.assertEqual(a['status']=='SAT',count_solutions(p,4)>0)
            with patch.object(search,'order_learned',side_effect=lambda model,g,d,c,values,steps: sorted(values)):
                b=solve(p,4,model=object(),trace=True)
            self.assertEqual(a['trace'],b['trace']); self.assertEqual(a['nodes'],b['nodes'])

    def test_model_order_is_permutation(self):
        from .search import order_learned
        from .soft import CardinalityBP
        from .graph import build_standard
        from .domains import init_domains
        g=build_standard(4)
        values=order_learned(CardinalityBP(60103),g,init_domains(g,[-1]*16),0,{0,1,2,3},1)
        self.assertEqual(sorted(values),list(range(4)))
