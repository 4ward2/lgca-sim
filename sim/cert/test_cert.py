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


def expected_verdict(task, seed):
    return 'PASS'


def verdict(t, task):
    return certificate(t) if task == 'M' else decide(t, task)


def fixture(task, seed):
    import json
    return json.loads((Path(__file__).parent / 'fixtures' / f'{task}_B_s{seed}_tables.json').read_text())


def dataset(task, n):
    if task != 'M':
        k = 2 if task == 'P' else 4
        a = np.array(list(itertools.product(range(k), repeat=n)))
        if task == 'P':
            return a + 2, np.cumsum(a, axis=1) % 2
        a = np.concatenate((a, np.zeros((len(a), 1), dtype=int)), axis=1)
        value = ((a & 1) * 2**np.arange(n+1)).sum(1) + (((a >> 1) & 1) * 2**np.arange(n+1)).sum(1)
        return a + 4, (value[:, None] // 2**np.arange(n+1)) % 2
    from .grid_exhaustive import bfs
    shape = (2, 3) if n == 6 else (1, n)
    a = np.array([x for x in itertools.product((4, 5, 7), repeat=n) if 7 in x]).reshape(-1, *shape)
    return a, np.array([bfs(x) for x in a])


def frozen_batch(t, x):
    import torch
    from sim.b_arm import TernaryNet
    m = TernaryNet(t['dims'], t['c'], t['V'])
    for key in ('u', 'r', 'o'):
        getattr(m, key).data.copy_(torch.tensor(t[key]))
    xi = torch.tensor(x)
    state = torch.ones((*xi.shape, t['V']))
    for rounds in range(t['V'] * int(np.prod(x.shape[1:])) + 1):
        nxt = m.update(state, xi)
        if torch.equal(state, nxt):
            return m.readout(state, xi).numpy().astype(int), rounds
        state = nxt
    raise AssertionError('no simulator stabilisation')


class CertificateTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('CERT_RUNS'), 'set CERT_RUNS to deployed run root')
    def test_deployed_runs(self):
        for task in ('P', 'A', 'M'):
            for seed in range(1, 11):
                _, _, t = load(Path(os.environ['CERT_RUNS']) / f'{task}_B_s{seed}')
                self.assertEqual(verdict(t, task)['verdict'], expected_verdict(task, seed))
                crosscheck(t, task, 80004)

    def test_portable_verdicts(self):
        for task, seed in itertools.product(('P', 'A', 'M'), range(1, 11)):
            r = verdict(fixture(task, seed), task)
            self.assertEqual(r['verdict'], expected_verdict(task, seed), (task, seed))
            self.assertEqual(len(r['checker_sha256']), 64)
            if task == 'M':
                x, y = dataset(task, 6)
                self.assertTrue(np.array_equal(frozen_batch(fixture(task, seed), x)[0], y))
            if task == 'M' and r['verdict'] == 'PASS':
                self.assertEqual(len(r['obligations']), 76 if seed in (1, 2, 6, 9, 10) else 75)
                self.assertTrue(all(o['passed'] for o in r['obligations']))
                from .grid import replay
                self.assertTrue(replay(fixture(task, seed), r['obligations']))
                self.assertFalse(replay(fixture(task, seed), r['obligations'][:-1]))

    def test_validation(self):
        from .tables import validate, validate_input
        for task in ('P', 'A', 'M'):
            t = extract(plant_ternary_oracle(task))
            for key, value in (('c', 0), ('dims', 3), ('V', 0), ('u', [[1]]), ('r', []), ('o', [[2]])):
                bad = copy.deepcopy(t); bad[key] = value
                with self.assertRaises(ValueError):
                    verdict(bad, task)
            for value in (0.5, 2, -1, '1'):
                bad = copy.deepcopy(t); bad['u'][0][0] = value
                with self.assertRaises(ValueError): validate(bad)
            with self.assertRaises(ValueError): validate(t, 'bad')
            with self.assertRaises(ValueError): validate_input(t, [0.5])
            with self.assertRaises(ValueError): validate_input(t, [])

    def test_halting_small_exhaustive(self):
        from .halting import decide as halt
        for task in ('P', 'A', 'M'):
            for seed in range(1, 11):
                t = fixture(task, seed)
                self.assertEqual(halt(t, task)['verdict'], 'UNKNOWN' if task == 'M' else 'PASS')
                for n in range(1, 5):
                    x, _ = dataset(task, n)
                    pred, rounds = frozen_batch(t, x)
                    self.assertTrue(np.array_equal(pred, dataset(task, n)[1]))
                    self.assertLessEqual(rounds, t['V'] * int(np.prod(x.shape[1:])))

    def test_adversarial_mutations(self):
        # All legal relation/unary/readout bits, including both virtual boundaries.
        # Bounded errors require FAIL and simulator witnesses; every surviving
        # mutation in this suite must also earn an all-size certificate PASS.
        summaries = []
        for task in ('P', 'A', 'M'):
            types = (2, 3) if task == 'P' else (4, 5, 6, 7) if task == 'A' else (4, 5, 7)
            deployed = [fixture(task, 3)] if task == 'M' else [fixture(task, 1)]
            for t in [extract(plant_ternary_oracle(task))] + deployed:
                for key in ('u', 'o'):
                    bad = copy.deepcopy(t); bad[key][0] = [1-v for v in bad[key][0]]
                    self.assertEqual(verdict(bad, task)['verdict'], 'PASS')
                counts = dict(PASS=0, FAIL=0, UNKNOWN=0, bounded_wrong=0)
                x, expected = dataset(task, 5 if task != 'M' else 6)
                indices = dict(u=[(a,h) for a in types for h in range(2)],
                    o=[(a,s) for a in types for s in range(4)],
                    r=[(e,a,b,h,j) for e in range(t['dims']) for a in (0,)+types
                       for b in (0,)+types if a or b for h in range(2) for j in range(2)])
                for key, inds in indices.items():
                    for idx in inds:
                        bad = copy.deepcopy(t); arr = np.array(bad[key]); arr[idx] ^= 1; bad[key] = arr.tolist()
                        pred, _ = frozen_batch(bad, x)
                        r = verdict(bad, task)
                        counts[r['verdict']] += 1
                        if not np.array_equal(pred, expected):
                            counts['bounded_wrong'] += 1
                            if task == 'M':
                                i = np.flatnonzero(np.any(pred != expected, axis=(1,2)))[0]
                                self.assertNotEqual(r['verdict'], 'PASS', (task,key,idx))
                                self.assertEqual(confirm(bad, x[i]), pred[i].tolist())
                                # Public bounded decision must promote the actual error to FAIL.
                                r = grid_decide(bad, witness=x[i])
                            self.assertEqual(r['verdict'], 'FAIL', (task,key,idx,r))
                            self.assertIn('simulator_prediction', r)
                        else:
                            self.assertEqual(r['verdict'], 'PASS', (task,key,idx,r))

                summaries.append(dict(task=task, counts=counts))
        print('MUTATION_COUNTS', summaries)

    def test_checker_mutant_is_killed(self):
        # Execute an actual source mutant in isolation: replace the backward
        # restriction by top. A relation mutation needs the right-to-left pass.
        import inspect
        import sim.cert.chain as chain_module
        source = inspect.getsource(chain_module).replace("t['o'][x][f & b]", "t['o'][x][f]")
        ns = dict(__name__='sim.cert._mutant', __package__='sim.cert')
        exec(compile(source, '<drop-backward-mutant>', 'exec'), ns)
        killed = False
        rng = np.random.default_rng(80005)
        for _ in range(200):
            bad = extract(plant_ternary_oracle('P'))
            for key in ('u', 'r', 'o'):
                bad[key] = rng.integers(0, 2, np.asarray(bad[key]).shape).tolist()
            try:
                mutant = ns['decide'](bad, 'P')
                real = decide(bad, 'P')
                killed |= mutant['verdict'] != real['verdict'] or mutant.get('counterexample') != real.get('counterexample')
            except AssertionError:
                killed = True
            if killed: break
        self.assertTrue(killed, 'suite failed to kill missing-backward checker')

    def test_oracles_and_corruption(self):
        for task in ('P', 'A', 'M'):
            t = extract(plant_ternary_oracle(task))
            crosscheck(t, task, 80002)
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
        rng = np.random.default_rng(80003)
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
