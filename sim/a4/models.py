"""Read-only adapters around existing model classes."""
import copy
import torch
from torch import nn
from sim.b_arm import BPNet, TernaryNet, bp_to_ternary
from sim.logic import LoopNet, FFNet, to_internal
from .protocol import check_seed

class UnsharedBPNet(nn.Module):
    def __init__(self, seed, depth=64):
        super().__init__()
        base=BPNet(1,1,K=2,seed=check_seed(seed))
        self.layers=nn.ModuleList([copy.deepcopy(base) for _ in range(depth)])
        self.O=nn.Parameter(base.O.detach().clone())
        for layer in self.layers:
            del layer._parameters['O']
            layer.register_buffer('O',self.O.detach().clone())

    def forward(self,xv,T=64,hard=False):
        if T!=len(self.layers): raise ValueError('unshared model has fixed trained depth')
        from sim.b_arm import xidx
        xi=xidx(xv); base=self.layers[0]
        M=xv.new_zeros((2,xv.shape[1],xv.shape[-1],2))
        for layer in self.layers: M=layer.messages(M,xi,hard)
        b=self.layers[-1].U[xi]+M.sum(0)
        return (torch.softmax(b,-1)*torch.sigmoid(self.O[xi])).sum(-1),M,b

def build(arm,seed):
    check_seed(seed)
    if arm in ('b','shared64'): return BPNet(1,1,K=2,seed=seed)
    if arm=='unshared64': return UnsharedBPNet(seed)
    if arm=='r': return LoopNet(1,1,G=384,seed=seed)
    if arm=='f': return FFNet(1,1,K=8,w=16,seed=seed)
    if arm=='dt-small':
        from .dt_adapter import build_dt
        return build_dt(seed)
    raise ValueError(arm)

def predict(model,arm,xv,T,hard=False):
    if arm in ('b','shared64','unshared64'): return model(xv,T,hard=hard)[0]
    return model(xv,T,mode='hard' if hard else 'soft')[0]

def oracle():
    t=TernaryNet(1,1,V=2)
    with torch.no_grad():
        t.o[:,2]=1
        for a in (2,3):
            for b in (2,3):
                for ha in range(2):
                    for hb in range(2): t.r[0,a,b,ha,hb]=int(hb==(ha^(b&1)))
        for b in (2,3):
            for hb in range(2): t.r[0,0,b,:,hb]=int(hb==(b&1))
    return t

def calibrate(bp,train):
    from .data import calibration
    fit=calibration(train)
    if not set(fit['sample_id']).issubset(set(train['sample_id'])): raise ValueError('calibration leakage')
    return bp_to_ternary(bp,fit,thr=2)

def calibration_audit(model,train):
    from .data import calibration
    from .evaluate import run_ac_to_fp
    from sim.b_arm import xidx
    import numpy as np
    fit=calibration(train); xv=to_internal(fit['x'])
    result=run_ac_to_fp(model,xv)
    states=(result['domains'][...,0]+2*result['domains'][...,1]).astype(int)
    codes=xidx(xv).numpy(); counts=np.zeros((4,4,2),dtype=int)
    for code in range(4):
        for state in range(4):
            labels=fit['y'][(codes==code)&(states==state)].astype(int)
            counts[code,state]=np.bincount(labels,minlength=2)
    return dict(sample_ids=fit['sample_id'].tolist(),counts=counts.tolist(),
                unseen=(counts.sum(-1)==0).tolist(),tie_policy='zero',threshold=2)
