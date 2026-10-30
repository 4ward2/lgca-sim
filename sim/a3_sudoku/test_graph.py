import unittest
from dataclasses import replace
from .graph import build_standard, build_samurai, validate_graph, canonical_factor_key


class GraphTests(unittest.TestCase):
    def test_sizes(self):
        for n in (4,9,16,25):
            g=build_standard(n)
            self.assertTrue(validate_graph(g))
            self.assertEqual(g.num_vars,n**3)
            self.assertEqual(len(g.groups),4*n*n)
            self.assertEqual(len(g.factor_vars),4*n**3)
    def test_samurai(self):
        g=build_samurai()
        self.assertEqual(len(g.cells),369)
        self.assertEqual(g.num_vars,3321)
        self.assertEqual(len(set(map(canonical_factor_key,g.groups))),len(g.groups))
        shared=g.cells.index((6,6))
        self.assertGreater(len(g.var_factors[g.var_ptr[shared*9]:g.var_ptr[shared*9+1]]),4)
        self.assertTrue(validate_graph(g))
    def test_reject(self):
        g=build_standard(4)
        for bad in (replace(g,factor_vars=(10000,)+g.factor_vars[1:]),
                    replace(g,var_ptr=(1,)+g.var_ptr[1:]),
                    replace(g,groups=g.groups+(g.groups[0],))):
            with self.assertRaises(ValueError):
                validate_graph(bad)
