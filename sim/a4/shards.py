"""Deterministic resumable evaluation chunks bound to frozen identities."""
from pathlib import Path
import numpy as np
from .protocol import write_json,read_json,digest,sha

def save_shard(root,start,pred,identity,diagnostics=None):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    path=root/f'{start:06d}.npz'; temporary=root/f'{start:06d}.tmp.npz'
    np.savez(temporary,pred=np.asarray(pred)); temporary.replace(path)
    write_json(path.with_suffix('.json'),dict(start=start,count=len(pred),identity=digest(identity),sha256=sha(path),diagnostics=diagnostics,diagnostics_sha256=digest(diagnostics)))

def load_shard(root,start,identity,count):
    path=Path(root)/f'{start:06d}.npz'
    if not path.exists(): return None
    record=read_json(path.with_suffix('.json'))
    if {k:record.get(k) for k in ('start','count','identity','sha256')}!=dict(start=start,count=count,identity=digest(identity),sha256=sha(path)):
        raise ValueError('shard identity/hash/count mismatch')
    if record.get('diagnostics_sha256')!=digest(record.get('diagnostics')):
        raise ValueError('shard diagnostics hash mismatch')
    with np.load(path,allow_pickle=False) as archive: pred=archive['pred']
    if len(pred)!=count or not np.isfinite(pred).all(): raise ValueError('invalid prediction shard')
    return pred

def validate_dag(plan):
    jobs=plan['jobs']; names=[j['name'] for j in jobs]
    if len(set(names))!=len(names): raise ValueError('duplicate job')
    visited=set()
    for job in jobs:
        if not set(job['needs']).issubset(visited): raise ValueError('missing/cyclic/forward dependency')
        name=job['name']; command=job['command']
        if f'--input in/{name} ' not in command or f'--out out/{name} ' not in command:
            raise ValueError('job artifact path contract')
        visited.add(name)
    return len(jobs)
