"""Real source mutants, executed in isolated module namespaces; never edit files."""
import copy
import inspect

import numpy as np

from . import chain, grid, tables


def mutant(module, before, after):
    source = inspect.getsource(module)
    assert source.count(before) >= 1, before
    source = source.replace(before, after)
    ns = dict(__name__='sim.cert._source_mutant', __package__='sim.cert', __file__=module.__file__)
    exec(compile(source, '<checker-source-mutant>', 'exec'), ns)
    return ns


def check_mutants(test):
    from .test_cert import fixture
    killed = []

    def rejects(name, probe):
        try:
            probe()
        except AssertionError:
            killed.append(name)
        else:
            test.fail('surviving checker source mutant: ' + name)

    # Compare domains AND outputs to the frozen independent simulator. Random
    # asymmetric tables distinguish axes/transposes; virtual u[0] is zeroed.
    rng = np.random.default_rng(90006)
    cases = []
    for _ in range(40):
        t = copy.deepcopy(fixture('M', 3))
        for key in ('u', 'r', 'o'):
            t[key] = (rng.random(np.asarray(t[key]).shape) < (0.8 if key == 'r' else 0.7)).astype(int).tolist()
        t['u'][0] = [0, 0]
        cases.append(t)
    for t in cases + [fixture('M', 3)]:
        test.assertEqual(tables.confirm(t, [[7, 5, 4], [5, 4, 5]]),
                         tables.interpret(t, [[7, 5, 4], [5, 4, 5]])[1].tolist())
    for name, before, after in (
        ('axis-swap', 'support(t, axis,', 'support(t, 1 - axis,'),
        ('missing-transposition', 'R[h][j]', 'R[j][h]'),
        ('virtual-cell-unary', 'else (0, full)', 'else (0, unary(t, 0))'),
        ('readout-bit-order', "[int(domains[p])]", "[int(domains[p]) // 2 + (int(domains[p]) % 2) * 2]"),
    ):
        ns = mutant(tables, before, after)
        def probe():
            for t in cases + [fixture('M', 3)]:
                x = [[7, 5, 4], [5, 4, 5]]
                real = tables.interpret(t, x)
                changed = ns['interpret'](t, x)
                # Baseline itself must agree with the frozen simulator.
                test.assertEqual(tables.confirm(t, x), real[1].tolist())
                test.assertEqual(changed[0].tolist(), real[0].tolist())
                test.assertEqual(changed[1].tolist(), real[1].tolist())
        rejects(name, probe)

    # Probe both A language end conditions, including actual readout-check
    # coverage when an extra illegal word happens to have the correct output.
    for name, before, after in (
        ('A-high-zero', "(alphabet if task == 'P' else (4,))", 'alphabet'),
        ('A-minimum-length', "(task == 'A' and ll == rl == 1)", 'False'),
    ):
        ns = mutant(chain, before, after)
        def probe():
            t = fixture('A', 1)
            if name == 'A-high-zero':
                for x in (5, 6, 7):
                    t['r'][0][x][0] = [[0, 0], [0, 0]]
                test.assertEqual(chain.decide(t, 'A')['verdict'], 'PASS')
                test.assertEqual(ns['decide'](t, 'A')['verdict'], 'PASS')
            else:
                # Count actual language/readout checks. Singleton [4] is
                # excluded even when its bit happens to match the legal words.
                class Readout(list):
                    def __getitem__(self, index):
                        calls.append(index)
                        return super().__getitem__(index)
                calls = []
                t['o'] = [Readout(row) for row in t['o']]
                chain.decide(t, 'A'); legal_checks = len(calls)
                calls.clear(); ns['decide'](t, 'A')
                test.assertEqual(len(calls), legal_checks)
        rejects(name, probe)

    ns = mutant(grid, '(True, False)', '(True,)')
    def probe():
        proof = ns['certificate'](fixture('M', 3))['obligations']
        # Coverage test requires every axis AND incoming direction; dropping a
        # whole direction otherwise leaves the remaining proof self-consistent.
        test.assertEqual({(o['axis'], o['target_right']) for o in proof if o['kind'] == 'support'},
                         {(0, True), (0, False), (1, True), (1, False)})
        test.assertTrue(grid.replay(fixture('M', 3), proof))
    rejects('dropped-M-direction', probe)

    ns = mutant(chain, "t['o'][x][f & b]", "t['o'][x][f]")
    def probe():
        for _ in range(200):
            t = copy.deepcopy(fixture('P', 1))
            for key in ('u', 'r', 'o'):
                t[key] = rng.integers(0, 2, np.asarray(t[key]).shape).tolist()
            real = chain.decide(t, 'P'); changed = ns['decide'](t, 'P')
            test.assertEqual((changed['verdict'], changed.get('counterexample')),
                             (real['verdict'], real.get('counterexample')))
    rejects('dropped-backward-pass', probe)
    print('CHECKER_SOURCE_MUTANTS_KILLED', killed)
