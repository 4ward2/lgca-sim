"""Independent integer-set semantics. Bit h denotes membership of latent h."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def validate(t, task=None):
    """Reject malformed public JSON tables before indexing or integer coercion."""
    if not isinstance(t, dict):
        raise ValueError('table must be a dictionary')
    if task is not None and task not in ('P', 'A', 'M'):
        raise ValueError('task must be P, A or M')
    for key in ('dims', 'c', 'V'):
        if type(t.get(key)) is not int or t[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if t['dims'] not in (1, 2):
        raise ValueError('dims must be 1 or 2')
    if task and (t['dims'], t['c']) != ((1, 1) if task == 'P' else (1, 2) if task == 'A' else (2, 2)):
        raise ValueError('task/c/dims mismatch')
    nin = 1 << (t['c'] + 1); v = t['V']
    shapes = dict(u=(nin, v), r=(t['dims'], nin, nin, v, v), o=(nin, 1 << v))
    for key, shape in shapes.items():
        try:
            a = np.asarray(t[key])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('invalid table ' + key) from exc
        if a.shape != shape or a.dtype.kind not in 'biuf' or not np.isin(a, (0, 1)).all():
            raise ValueError(f'{key} must have shape {shape} and only 0/1 values')
    return t


def provenance():
    import platform
    import torch
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob('*.py'))}
    return dict(checker_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                checker_sources=sources, python_version=platform.python_version(), torch_version=torch.__version__)


def validate_input(t, xi):
    validate(t)
    a = np.asarray(xi)
    if (a.ndim != t['dims'] or not a.size or any(n == 0 for n in a.shape)
            or a.dtype.kind not in 'iu' or (a < 0).any() or (a >= 1 << (t['c'] + 1)).any()):
        raise ValueError('input must be a nonempty integer array of matching dims and valid codes')
    return a


def extract(model, checkpoint=None):
    out = dict(dims=model.dims, c=model.c, V=model.V)
    for name in ('u', 'r', 'o'):
        a = getattr(model, name).detach().cpu().numpy()
        if not np.isin(a, [0, 1]).all():
            raise ValueError('non-bit table: ' + name)
        out[name] = a.astype(int).tolist()
    validate(out)
    out['checkpoint_sha256'] = (hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                                if checkpoint else None)
    out['table_sha256'] = hashlib.sha256(hashable(out).encode()).hexdigest()
    return out


def hashable(t):
    """Canonical immutable representation, excluding provenance."""
    return json.dumps({k: t[k] for k in ('dims', 'c', 'V', 'u', 'r', 'o')}, sort_keys=True)


def load(path):
    from sim import b_arm
    cfg, ck = b_arm.load_run(str(path))
    m = ck['best_deploy']['ternary']
    return cfg, m, extract(m, Path(path) / 'best_deploy.pt')


def unary(t, x):
    return sum(int(v) << h for h, v in enumerate(t['u'][x]))


def support(t, e, a, b, domain, target_right=True):
    R = t['r'][e][a][b]
    V = t['V']
    return sum((any(domain & (1 << j) and (R[j][h] if target_right else R[h][j])
                    for j in range(V))) << h for h in range(V))


def interpret(t, xi):
    """Synchronous from full domains; virtual input 0/full is re-created each round.

    At most V*N strict rounds occur: monotonicity follows by induction from
    top, and each strict round deletes at least one of the V*N bits.
    """
    xi = validate_input(t, xi)
    full = (1 << t['V']) - 1
    domains = np.full(xi.shape, full, dtype=int)
    for rounds in range(t['V'] * xi.size + 1):
        nxt = np.empty_like(domains)
        for p in np.ndindex(xi.shape):
            x = int(xi[p]); s = unary(t, x)
            for axis, delta in itertools.product(range(xi.ndim), (-1, 1)):
                q = list(p); q[axis] += delta; q = tuple(q)
                inside = all(0 <= q[k] < xi.shape[k] for k in range(xi.ndim))
                y, d = (int(xi[q]), int(domains[q])) if inside else (0, full)
                if delta == -1:
                    s &= support(t, axis, y, x, d, True)
                else:
                    s &= support(t, axis, x, y, d, False)
            nxt[p] = s
        if np.array_equal(nxt, domains):
            pred = np.array([t['o'][int(xi[p])][int(domains[p])] for p in np.ndindex(xi.shape)])
            return domains, pred.reshape(xi.shape), rounds
        domains = nxt
    raise AssertionError('monotone convergence bound violated')


def confirm(t, xi):
    """Cross-check sets AND readout using the frozen simulator, not this interpreter."""
    xi = validate_input(t, xi)
    import torch
    from sim.b_arm import TernaryNet
    m = TernaryNet(t['dims'], t['c'], t['V'])
    for name in ('u', 'r', 'o'):
        getattr(m, name).data.copy_(torch.tensor(t[name]))
    xi = np.asarray(xi, dtype=int)
    xt = torch.tensor(xi[None])
    c = torch.ones(tuple(xt.shape) + (t['V'],))
    for _ in range(t['V'] * xi.size + 1):
        nxt = m.update(c, xt)
        if torch.equal(c, nxt):
            break
        c = nxt
    else:
        raise AssertionError('simulator did not stabilise')
    assert torch.equal(c, m.update(c, xt)), 'simulator did not stabilise'
    s = m.subset_index(c)[0].numpy()
    y = m.readout(c, xt)[0].numpy().astype(int)
    rs, ry, _ = interpret(t, xi)
    assert np.array_equal(s, rs) and np.array_equal(y, ry)
    return y.tolist()
