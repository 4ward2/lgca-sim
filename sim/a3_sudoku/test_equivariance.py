import random
import unittest
from dataclasses import replace
from .graph import build_standard, build_samurai, validate_graph
from .domains import EO_RELATION, init_domains, synchronous_closure, read_solution


class Equivariance(unittest.TestCase):
    def test_legal_permutations(self):
        rng=random.Random(60002)
        for n in (4,9):
            b=int(n**.5)
            g=build_standard(n)
            sol=[(r*b+r//b+c)%n for r in range(n) for c in range(n)]
            p=[v if rng.random()<.4 else -1 for v in sol]
            base=synchronous_closure(g,init_domains(g,p))
            digits=list(range(n)); rng.shuffle(digits)
            def axis():
                blocks=list(range(b)); rng.shuffle(blocks)
                out=[]
                for block in blocks:
                    within=list(range(b)); rng.shuffle(within)
                    out.extend(block*b+i for i in within)
                return out
            rows,cols=axis(),axis()
            for transpose in (False,True):
                mapping=[(cols[c]*n+rows[r] if transpose else rows[r]*n+cols[c]) for r in range(n) for c in range(n)]
                q=[-1]*(n*n)
                for c,v in enumerate(p):
                    q[mapping[c]]=-1 if v==-1 else digits[v]
                result=synchronous_closure(g,init_domains(g,q))
                for c in range(n*n):
                    for v in range(n):
                        self.assertEqual(result[mapping[c]*n+digits[v]],base[c*n+v])
    def test_fixed_table_sizes(self):
        table=bytes(EO_RELATION)
        for n in (4,9,16,25):
            g=build_standard(n)
            p=[-1]*(n*n); p[0]=0
            d=synchronous_closure(g,init_domains(g,p),relation=EO_RELATION)
            self.assertEqual(d[0],2)
            self.assertEqual(d[n],1)
            self.assertEqual(bytes(EO_RELATION),table)
            self.assertIsNone(read_solution(g,d,p))
    def test_shared_cell_propagation(self):
        g=build_samurai()
        p=[-1]*369
        c=g.cells.index((6,6)); p[c]=3
        d=synchronous_closure(g,init_domains(g,p))
        # Both top-left and central board rows receive this shared given.
        for xy in ((6,0),(6,14),(0,6),(14,6)):
            self.assertEqual(d[g.cells.index(xy)*9+3],1)
        self.assertEqual(d[c*9+3],2)
    def test_reject_malformed(self):
        g=build_standard(4)
        for bad in (replace(g,regions=((0,0,1,2),)),replace(g,factor_type=('BAD',)+g.factor_type[1:])):
            with self.assertRaises(ValueError):
                validate_graph(bad)
    def test_unconnected_components(self):
        from .graph import _make
        from .oracle import standard_regions
        n=4
        regions=[(tuple(c+offset for c in r),'ROW_VALUE') for offset in (0,16) for r in standard_regions(4)]
        g=_make(n,tuple(range(32)),regions,[tuple(range(16)),tuple(range(16,32))])
        p=[-1]*32; p[0]=0
        d=synchronous_closure(g,init_domains(g,p))
        self.assertEqual(d[64:],[3]*64)
