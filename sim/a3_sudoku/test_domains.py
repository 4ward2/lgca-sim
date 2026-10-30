import itertools
import random
import unittest
from .domains import factor_support, init_domains, synchronous_closure, candidate_sets
from .graph import build_standard
from .oracle import singles_closure


def brute_support(ds,accepted=(1,)):
    out=[]
    for i in range(len(ds)):
        mask=0
        for h in (0,1):
            others=ds[:i]+ds[i+1:]
            for zs in itertools.product((0,1),repeat=len(others)):
                if min(2,h+sum(zs)) in accepted and all(d & (1<<z) for d,z in zip(others,zs)):
                    mask|=1<<h
                    break
        out.append(mask)
    return out


class Domains(unittest.TestCase):
    def test_exhaustive(self):
        for m in range(1,7):
            for ds in itertools.product(range(4),repeat=m):
                self.assertEqual(factor_support(ds),brute_support(ds))
    def test_closures(self):
        rng=random.Random(60001)
        for n in (4,9):
            b=int(n**.5)
            sol=[(r*b+r//b+c)%n for r in range(n) for c in range(n)]
            g=build_standard(n)
            for _ in range(12):
                p=[v if rng.random()<.5 else -1 for v in sol]
                old=init_domains(g,p)
                sync=synchronous_closure(g,old)
                self.assertTrue(all(a & b==a for a,b in zip(sync,old)))
                self.assertEqual(candidate_sets(g,sync),singles_closure(p,n))
                async_d=old[:]
                while True:
                    before=async_d[:]
                    order=list(range(len(g.groups)))
                    rng.shuffle(order)
                    for f in order:
                        group=g.groups[f]
                        support=factor_support([async_d[v] for v in group])
                        for v,s in zip(group,support):
                            async_d[v]&=s
                    if async_d==before:
                        break
                self.assertEqual(async_d,sync)
    def test_initial_validation(self):
        with self.assertRaises(ValueError):
            init_domains(build_standard(4),[0,0]+[-1]*14)
        for ds in ([0,3],[2,2],[1,1]):
            self.assertEqual(factor_support(ds),brute_support(ds))
    def test_general_relations(self):
        for relation in itertools.product((False,True),repeat=3):
            accepted=tuple(k for k,x in enumerate(relation) if x)
            for m in range(1,4):
                for ds in itertools.product(range(4),repeat=m):
                    self.assertEqual(factor_support(ds,relation),brute_support(ds,accepted))
