"""Light local checks; no training and no locked seeds."""
import copy
import itertools
import os
from pathlib import Path
import unittest

import numpy as np

from sim.b_arm import plant_ternary_oracle
from .tables import extract, interpret, confirm, unary, support, load
from .chain import decide
from .grid import certificate, decide as grid_decide
from .grid_exhaustive import search
from .run_all import crosscheck, equivalence_key


class CertificateTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('CERT_RUNS'), 'set CERT_RUNS to deployed run root')
    def test_deployed_runs(self):
        for task in ('P', 'A', 'M'):
            _, _, t = load(Path(os.environ['CERT_RUNS']) / f'{task}_B_s1')
            crosscheck(t, task, 50004)

    def test_oracles_and_corruption(self):
        for task in ('P', 'A', 'M'):
            t = extract(plant_ternary_oracle(task))
            crosscheck(t, task, 50002)
            result = certificate(t) if task == 'M' else decide(t, task)
            self.assertEqual(result['verdict'], 'PASS', (task, result))
            bad = copy.deepcopy(t)
            # A relevant singleton readout, one bit only.
            x = 2 if task == 'P' else (4 if task == 'A' else 7)
            s = 2 if task == 'M' else 1
            bad['o'][x][s] ^= 1
            result = grid_decide(bad) if task == 'M' else decide(bad, task)
            self.assertEqual(result['verdict'], 'FAIL', (task, result))
            self.assertIn('simulator_prediction', result)

    def test_maze_exhaustive(self):
        t = extract(plant_ternary_oracle('M'))
        r = search(t, 6, 30)
        self.assertEqual(r['verdict'], 'BOUNDED_PASS', r)

    def test_chain_arbitrary_tables(self):
        # Exercise contradictions and non-oracle relations, not just learned tables.
        rng = np.random.default_rng(50003)
        for task in ('P', 'A'):
            for _ in range(8):
                t = extract(plant_ternary_oracle(task))
                for name in ('u', 'r', 'o'):
                    a = np.array(t[name]); t[name] = rng.integers(0, 2, a.shape).tolist()
                verdict = decide(t, task)
                self.assertIn(verdict['verdict'], ('PASS', 'FAIL'))
                shortest = None
                for length in range(1, 4):
                    alphabet = (2, 3) if task == 'P' else (4, 5, 6, 7)
                    for word in itertools.product(alphabet, repeat=length):
                        confirm(t, word)
                        f = []; b = [0] * length
                        for i, x in enumerate(word):
                            f.append(unary(t, x) & support(t, 0, word[i-1] if i else 0, x,
                                                          f[-1] if i else 3))
                        for i in reversed(range(length)):
                            b[i] = unary(t, word[i]) & support(t, 0, word[i],
                                word[i+1] if i+1 < length else 0,
                                b[i+1] if i+1 < length else 3, False)
                        states, pred, _ = interpret(t, word)
                        self.assertEqual(states.tolist(), [a & z for a, z in zip(f, b)])
                        if task == 'A' and (length < 2 or word[-1] != 4):
                            continue
                        st = 0; expected = []
                        for x in word:
                            z = (st ^ (x & 1)) if task == 'P' else st + (x & 1) + ((x >> 1) & 1)
                            expected.append(z & 1)
                            st = z if task == 'P' else z >> 1
                        if pred.tolist() != expected and shortest is None:
                            shortest = length
                if shortest is not None:
                    self.assertEqual(verdict['verdict'], 'FAIL')
                    self.assertEqual(len(verdict['counterexample']), shortest)

    def test_equivalence_relabelling(self):
        for task in ('P', 'A', 'M'):
            t = extract(plant_ternary_oracle(task)); other = copy.deepcopy(t)
            x = 3 if task == 'P' else 5
            other['u'][x] = other['u'][x][::-1]
            other['o'][x][1], other['o'][x][2] = other['o'][x][2], other['o'][x][1]
            r = np.array(other['r'])
            r[:, x] = r[:, x, :, ::-1, :]
            r[:, :, x] = r[:, :, x, :, ::-1]
            other['r'] = r.tolist()
            self.assertEqual(equivalence_key(t, task), equivalence_key(other, task))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    unittest.main()
