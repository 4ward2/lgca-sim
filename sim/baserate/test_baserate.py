"""Fast tests for the base-rate control (no training)."""
import copy
import tempfile
import unittest

import numpy as np

from sim.b_arm import plant_ternary_oracle
from sim.cert.tables import extract, validate
from sim.baserate import baserate as B

SHAPES = {'P': dict(dims=1, c=1), 'A': dict(dims=1, c=2), 'M': dict(dims=2, c=2)}


class BaseRateTests(unittest.TestCase):
    def test_planted_oracles_pass(self):
        for task in 'PAM':
            t = extract(plant_ternary_oracle(task))
            cat, _ = B.classify(t, task)
            self.assertEqual(cat, 'PASS', task)

    def test_corrupted_table_fails(self):
        for task in 'PA':
            t = extract(plant_ternary_oracle(task))
            t = copy.deepcopy(t)
            # flip a readout bit that the oracle relies on
            t['o'][2 if task == 'P' else 4][2] ^= 1
            cat, det = B.classify(t, task)
            self.assertEqual(cat, 'FAIL', task)
            self.assertTrue(det['cx'])
        t = copy.deepcopy(extract(plant_ternary_oracle('M')))
        t['o'][5][2] ^= 1  # open cell fully resolved no longer reads 1
        cat, det = B.classify(t, 'M')
        self.assertEqual(cat, 'UNKNOWN')
        self.assertEqual(det['bounded'], 'FAIL')

    def test_sampler_shapes_and_densities(self):
        rng = np.random.default_rng(800001)
        for task in 'PAM':
            reals = B.real_tables(task)
            ref = reals[0]
            for t in (B.sample_uniform(task, rng), B.sample_density(task, rng, reals)):
                validate(t, task)
                for k in ('u', 'r', 'o'):
                    self.assertEqual(np.shape(t[k]), np.shape(ref[k]))
                self.assertEqual((t['dims'], t['c'], t['V']), (ref['dims'], ref['c'], 2))
            # density: each slice's popcount is that of the same slice in some real table
            for _ in range(5):
                t = B.sample_density(task, rng, reals)
                for key, sd in (('u', 1), ('r', 2), ('o', 1)):
                    mine = np.asarray(t[key]); stack = np.stack([np.asarray(r[key]) for r in reals])
                    ax = tuple(range(mine.ndim - sd, mine.ndim))
                    sums = mine.sum(axis=ax)
                    real_sums = stack.sum(axis=tuple(a + 1 for a in ax))
                    for idx in np.ndindex(sums.shape):
                        self.assertIn(sums[idx], set(real_sums[(slice(None),) + idx]))
            # uniform density ~ 1/2 over many samples
            m = np.mean([np.mean(B.sample_uniform(task, rng)['r']) for _ in range(30)])
            self.assertAlmostEqual(m, 0.5, delta=0.05)

    def test_end_to_end_small(self):
        for task in 'PAM':
            with tempfile.TemporaryDirectory() as d:
                res = B.run(task, 'density', 5, 800003, d, 60)
                self.assertEqual(res['tested'], 5)
                self.assertEqual(res['pass_count'], 0)

    def test_wilson(self):
        self.assertAlmostEqual(B.wilson_upper(0, 100000), 3.84e-5, delta=0.02e-5)

    def test_real_zero_prop_fixed_point_matches(self):
        rows = B.zero_prop('P', 800006, n_samples=40)
        for r in rows:
            self.assertEqual(r['fixed_point']['exact_match'], 1.0)


if __name__ == '__main__':
    unittest.main()
