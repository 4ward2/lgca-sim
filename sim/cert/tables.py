"""Independent integer-set semantics. Bit h denotes membership of latent h."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def extract(model, checkpoint=None):
    out = dict(dims=model.dims, c=model.c, V=model.V)
    for name in ('u', 'r', 'o'):
        a = getattr(model, name).detach().cpu().numpy()
        if not np.isin(a, [0, 1]).all():
            raise ValueError('non-bit table: ' + name)
        out[name] = a.astype(int).tolist()
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
    xi = np.asarray(xi, dtype=int)
    assert xi.ndim == t['dims']
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
    s = m.subset_index(c)[0].numpy()
    y = m.readout(c, xt)[0].numpy().astype(int)
    rs, ry, _ = interpret(t, xi)
    assert np.array_equal(s, rs) and np.array_equal(y, ry)
    return y.tolist()
