"""Public tensor format, isolated train/dev artifacts and reversible encodings."""
from pathlib import Path
import numpy as np
import torch
from .protocol import check_seed, sha

def make_official_split(count=10000):
    if count != 10000: raise ValueError('official pool must contain 10000 examples')
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(42)).numpy()
    return indices[:8000], indices[8000:]

def normalize_roundtrip(loader_x):
    x = np.asarray(loader_x, dtype=np.float32)+.5
    if not np.isin(x,[0,1]).all(): raise ValueError('nonbinary loader input')
    return x

def load_prefix(root, length, *, role, expected_hashes=None, fixture=False):
    root = Path(root)
    if role not in ('train','dev','test'): raise ValueError('unknown role')
    if role != 'test' and length != 32 and not fixture: raise ValueError('training cannot access OOD')
    folder = root/'prefix_sums_data' if (root/'prefix_sums_data').exists() else root
    paths = [folder/f'{length}_{kind}.pth' for kind in ('data','targets')]
    if expected_hashes:
        for p in paths:
            if sha(p) != expected_hashes[p.name]: raise ValueError('data hash changed')
    x,y = [torch.load(p,weights_only=True,map_location='cpu').numpy() for p in paths]
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != length or (not fixture and len(x)!=10000):
        raise ValueError('unexpected upstream tensor shape')
    if not np.isin(x,[0,1]).all() or not np.array_equal(y,np.cumsum(x,axis=1)%2):
        raise ValueError('invalid upstream bits or targets')
    ids = np.arange(len(x))
    if role != 'test' and not fixture:
        ids = make_official_split()[0 if role=='train' else 1]
    return dict(x=x[ids,None].astype(np.float32), y=y[ids].astype(np.int64),
                mask=np.ones_like(y[ids],dtype=bool), diff=np.full(len(ids),length), sample_id=ids)

def regenerate_prefix(length, seed, count=10000):
    check_seed(seed,'data')
    if length<24: raise ValueError('A4 regeneration only supports L>=24')
    x=(torch.rand(count,length,generator=torch.Generator().manual_seed(seed))>=.5).float()
    return x, x.cumsum(1).remainder(2).long()

def calibration(train):
    ids=np.argsort(train['sample_id'])[:128]
    return {k:v[ids] for k,v in train.items()}

def decode_maze(rgb, target):
    rgb=np.asarray(rgb); target=np.asarray(target)
    if rgb.ndim!=3 or rgb.shape[0]!=3 or rgb.shape[1:]!=target.shape or not np.isin(rgb,[0,1]).all():
        raise ValueError('maze schema')
    if np.any(rgb[:,[0,-1],:]) or np.any(rgb[:,:,[0,-1]]) or np.any(target[[0,-1],:]) or np.any(target[:,[0,-1]]):
        raise ValueError('nonzero outside border')
    def shrink(a):
        core=a[...,1:-1,1:-1]
        small=core[...,::2,::2]
        if not np.array_equal(core,np.repeat(np.repeat(small,2,-2),2,-1)):
            raise ValueError('not uniform 2x2 blocks')
        return small
    small=shrink(rgb); mask=shrink(target)
    source=np.all(small==np.array([0,1,0])[:,None,None],axis=0)
    goal=np.all(small==np.array([1,0,0])[:,None,None],axis=0)
    if source.sum()!=1 or goal.sum()!=1: raise ValueError('endpoint count')
    return dict(rgb=small, open=small.max(0)>0, source=source, goal=goal, target=mask)

def encode_maze_mask(mask):
    a=np.repeat(np.repeat(np.asarray(mask),2,-2),2,-1)
    return np.pad(a,[(0,0)]*(a.ndim-2)+[(1,1),(1,1)])
