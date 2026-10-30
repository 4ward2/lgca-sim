import numpy as np
from .protocol import SEEDS

def score_exact(pred,target):
    if not np.isfinite(np.asarray(pred)).all(): raise ValueError('nonfinite prediction')
    pred=np.asarray(pred)>.5; target=np.asarray(target).astype(bool)
    if pred.shape!=target.shape or not pred.size: raise ValueError('prediction shape')
    return dict(exact=float((pred==target).reshape(len(pred),-1).all(1).mean()),bitacc=float((pred==target).mean()),count=len(pred))

def aggregate_seeds(records,expected=SEEDS):
    if len(records)!=len(expected) or {r['seed'] for r in records}!=set(expected): raise ValueError('missing/duplicate seed')
    a=np.array([r['exact'] for r in records])
    if not np.isfinite(a).all(): raise ValueError('nonfinite score')
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else None,median=float(np.median(a)),min=float(a.min()),max=float(a.max()))

def paired_bootstrap(control,intervention,draws=10000):
    """Resample training seeds, after averaging random draws within parents."""
    a=np.asarray(control,dtype=float); b=np.asarray(intervention,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)!=10 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired comparison requires ten finite seed values')
    rng=np.random.default_rng(173901)
    difference=b-a
    boot=difference[rng.integers(0,10,size=(draws,10))].mean(1)
    return dict(difference=float(difference.mean()),per_seed=difference.tolist(),ci95=np.quantile(boot,[.025,.975]).tolist(),draws=draws)

def holm(pvalues):
    values=np.asarray(pvalues,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or ((values<0)|(values>1)).any(): raise ValueError('invalid p-values')
    order=np.argsort(values); adjusted=np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values))))
    result=np.empty_like(values); result[order]=np.minimum(adjusted,1)
    return result.tolist()
