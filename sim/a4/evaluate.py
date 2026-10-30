"""Whole-state fixed points and streaming continuous inference."""
import hashlib
import numpy as np
import torch
from sim.b_arm import xidx
from .protocol import OBSERVE

@torch.no_grad()
def run_ac_to_fp(model,xv,trace=False):
    xi=xidx(xv); c=torch.ones((*xi.shape,2),device=xv.device)
    n=xi.shape[-1]; times=torch.zeros(len(xi),dtype=torch.long)
    history=[c.clone()] if trace else None
    curve={}; hashes=[]
    for step in range(1,2*n+2):
        nxt=model.update(c,xi)
        if not torch.all(nxt<=c): raise AssertionError('domains expanded')
        same=(nxt==c).flatten(1).all(1)
        times[(times==0)&same]=step
        c=nxt
        if trace: history.append(c.clone())
        hashes.append(hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest())
        if step in OBSERVE or step==2*n+1: curve[step]=model.readout(c,xi).cpu().numpy()
        if torch.all(times>0): break
    if not torch.all(times>0): raise AssertionError('fixed point bound violated')
    final=model.readout(c,xi).cpu().numpy()
    for step in set([v for v in OBSERVE if v<=2*n+1]+[2*n+1]): curve.setdefault(step,final)
    return dict(pred=final,domains=c.cpu().numpy(),t_fp=times.numpy(),curve=curve,state_hashes=hashes,trajectory=history)

@torch.no_grad()
def run_depth_bp(model,xv,T):
    """Read each trained layer using the common final readout."""
    if not hasattr(model,'layers'): return run_bp(model,xv,T)
    if T!=len(model.layers): raise ValueError('unshared model has fixed trained depth')
    xi=xidx(xv); M=xv.new_zeros((2,xv.shape[1],xv.shape[-1],2)); curve={}
    for step,layer in enumerate(model.layers,1):
        M=layer.messages(M,xi,False)
        if step in OBSERVE or step==T:
            b=layer.U[xi]+M.sum(0)
            curve[step]=(torch.softmax(b,-1)*torch.sigmoid(model.O[xi])).sum(-1).cpu().numpy()
    return dict(pred=curve[T],curve=curve)

@torch.no_grad()
def run_bp(model,xv,T,hard=False):
    xi=xidx(xv); M=xv.new_zeros((2,xv.shape[1],xv.shape[-1],2)); curve={}
    for step in range(1,T+1):
        M=model.messages(M,xi,hard)
        if step in OBSERVE or step==T: curve[step]=model.belief(M,xi,hard)[0].cpu().numpy()
    return dict(pred=curve[T],curve=curve)

@torch.no_grad()
def run_dt_streaming(model,x,T,max_conf=False):
    h=model.projection(x); chosen=None; confidence=None; curve={}
    for step in range(1,T+1):
        h=model.recur_block(torch.cat([h,x],1)); logits=model.head(h)
        pred=logits.argmax(1); conf=logits.softmax(1).amax(1).sum(1)
        if max_conf:
            update=torch.ones_like(conf,dtype=torch.bool) if confidence is None else conf>=confidence
            chosen=pred.clone() if chosen is None else torch.where(update[:,None],pred,chosen)
            confidence=conf if confidence is None else torch.maximum(confidence,conf)
        if step in OBSERVE or step==T: curve[step]=(chosen if max_conf else pred).cpu().numpy()
    return dict(pred=curve[T],curve=curve)

@torch.no_grad()
def run_logic(model,xv,T):
    """R is recurrent; F has exactly its trained block stack."""
    if hasattr(model,'blocks'):
        h=xv.new_zeros((model.w,)+tuple(xv.shape[1:])); curve={}
        for step,block in enumerate(model.blocks,1):
            h=block(h,xv,'hard')
            if step in OBSERVE: curve[step]=h[0].cpu().numpy()
        final=h[0].cpu().numpy()
        for step in [v for v in OBSERVE if v<=T]+[T]: curve.setdefault(step,final)
        return dict(pred=final,curve=curve)
    h=model.init_state(xv); curve={}
    for step in range(1,T+1):
        h=model.step(h,xv,'hard')
        if step in OBSERVE or step==T: curve[step]=h[0].cpu().numpy()
    return dict(pred=curve[T],curve=curve)
