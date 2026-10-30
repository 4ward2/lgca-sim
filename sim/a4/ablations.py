import copy
import numpy as np
import torch
from .protocol import check_seed, sha
from .models import calibrate

def deploy_variants(bp,train,checkpoint):
    soft=copy.deepcopy(bp); soft.infer='sum'
    qbp=copy.deepcopy(bp); qbp.infer='max'; qbp.Q=6; qbp.delta=.25
    return dict(soft=soft,qbp=qbp,ac=calibrate(bp,train),source_checkpoint_sha256=sha(checkpoint))

def flip_unknown(model):
    out=copy.deepcopy(model)
    with torch.no_grad(): out.o[2:4,3]=1-out.o[2:4,3]
    return out

def randomize_equal_density(model,seed,train):
    check_seed(seed,'density'); rng=np.random.default_rng(seed); out=copy.deepcopy(model)
    with torch.no_grad():
        for name in ('u','r'):
            a=getattr(out,name); width=2 if name=='u' else 4
            for row in a.reshape(-1,width): row.copy_(row[torch.tensor(rng.permutation(width))].clone())
        for row in out.o: row[1:3]=row[1:3][torch.tensor(rng.permutation(2))].clone()
        out.o[:,[0,3]]=0
        from .data import calibration
        from .evaluate import run_ac_to_fp
        from sim.logic import to_internal
        from sim.b_arm import xidx
        fit=calibration(train); xv=to_internal(fit['x']); result=run_ac_to_fp(out,xv)
        states=result['domains'][...,0]+2*result['domains'][...,1]; xi=xidx(xv).numpy()
        for x in range(4):
            for s in (0,3):
                labels=fit['y'][(xi==x)&(states==s)]
                out.o[x,s]=int(len(labels)>0 and labels.sum()>len(labels)/2)
    return out
