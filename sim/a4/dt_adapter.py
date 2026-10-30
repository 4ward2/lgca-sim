"""Pinned MIT upstream implementation; only streaming evaluation is adapted.

Sources are retained byte-for-byte as .txt to avoid importing upstream packages.
No dependency installation or GPU probing occurs here.
"""
from pathlib import Path
import types
import sys
import random
import torch
from .protocol import DT_SHA, check_seed, sha,read_json

def upstream_module():
    root=Path(__file__).parent
    pinned=read_json(root/'upstream_sources.json')
    if pinned['commit']!=DT_SHA or any(sha(root/name)!=expected for name,expected in pinned['files'].items()):
        raise ValueError('pinned upstream source bytes changed')
    package=__name__+'.pinned'
    if package+'.dt' in sys.modules: return sys.modules[package+'.dt']
    pkg=types.ModuleType(package); pkg.__path__=[]; sys.modules[package]=pkg
    for name,file in [('blocks','upstream_blocks.txt'),('dt','upstream_dt_net_1d.txt')]:
        module=types.ModuleType(package+'.'+name); module.__package__=package
        sys.modules[module.__name__]=module
        exec(compile((root/file).read_text(encoding='utf-8'),str(root/file),'exec'),module.__dict__)
    return module

def build_dt(seed):
    check_seed(seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return upstream_module().dt_net_recall_1d(width=32).cpu()

def progressive(model,x,rng,max_iters=30):
    n=rng.randrange(0,max_iters); k=rng.randrange(1,max_iters-n+1)
    interim=None
    if n>0:
        with torch.no_grad(): _,interim=model(x,iters_to_do=n)
        interim=interim.detach()
    outputs,_=model(x,iters_elapsed=n,iters_to_do=k,interim_thought=interim)
    return outputs,k

def source_manifest():
    root=Path(__file__).parent
    return dict(commit=DT_SHA,license='MIT',ported=False,
                files={p.name:sha(p) for p in root.glob('upstream_*.txt')})
