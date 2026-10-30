"""Differentiable saturated-count sum-product BP; no training entry point."""
import torch
from torch import nn
from .oracle import check_puzzle


def _lse(values,dim=0):
    """All -inf has value -inf and zero gradient, rather than NaN."""
    any_finite=torch.isfinite(values).any(dim=dim,keepdim=True)
    safe=torch.where(any_finite,values,torch.zeros_like(values))
    result=torch.logsumexp(safe,dim=dim)
    return torch.where(any_finite.squeeze(dim),result,torch.full_like(result,-torch.inf))


def normalize(message):
    z=_lse(message,-1).unsqueeze(-1)
    safe_z=torch.where(torch.isfinite(z),z,torch.zeros_like(z))
    return message-safe_z


def log_count_convolve(a,b):
    """Three slots represent counts 0, 1, >=2."""
    return torch.stack([_lse(torch.stack([a[i]+b[j] for i in range(3) for j in range(3)
                                         if min(2,i+j)==k])) for k in range(3)])


def factor_messages(incoming,theta):
    if incoming.ndim!=2 or incoming.shape[1]!=2 or theta.shape!=(3,):
        raise ValueError('expected [m,2] messages and three potentials')
    identity=incoming.new_tensor([0.,-torch.inf,-torch.inf])
    leaves=[torch.stack((row[0],row[1],row.new_tensor(-torch.inf))) for row in incoming]
    prefix=[identity]
    for leaf in leaves:
        prefix.append(log_count_convolve(prefix[-1],leaf))
    suffix=[identity]*(len(leaves)+1)
    for i in range(len(leaves)-1,-1,-1):
        suffix[i]=log_count_convolve(leaves[i],suffix[i+1])
    rows=[]
    for i in range(len(leaves)):
        other=log_count_convolve(prefix[i],suffix[i+1])
        rows.append(torch.stack([_lse(torch.stack([other[k]+theta[min(2,k+h)] for k in range(3)]))
                                 for h in (0,1)]))
    return normalize(torch.stack(rows)) if rows else incoming.clone()


def _excluded_sums(values,zero):
    prefix=[zero]
    for value in values:
        prefix.append(prefix[-1]+value)
    suffix=[zero]*(len(values)+1)
    for i in range(len(values)-1,-1,-1):
        suffix[i]=values[i]+suffix[i+1]
    return [prefix[i]+suffix[i+1] for i in range(len(values))],prefix[-1]


class CardinalityBP(nn.Module):
    def __init__(self,seed=60000):
        super().__init__()
        if seed<60000:
            raise ValueError('A3 development seeds must be >=60000')
        rng=torch.Generator().manual_seed(seed)
        self.theta=nn.Parameter(.1*torch.randn(3,generator=rng))
        self.beta=nn.Parameter(.1*torch.randn(2,generator=rng))

    def forward(self,graph,givens,steps=1,return_history=False):
        check_puzzle(givens,graph.n,graph.regions)
        if steps<0:
            raise ValueError('negative steps')
        unary=self.beta.expand(graph.num_vars,2)
        allowed=torch.ones_like(unary,dtype=torch.bool)
        for c,v in enumerate(givens):
            if v>=0:
                for a in range(graph.n):
                    allowed[c*graph.n+a,0 if a==v else 1]=False
        unary=torch.where(allowed,unary,torch.full_like(unary,-torch.inf))
        edges=[]
        adjacency=[[] for _ in range(graph.num_vars)]
        for group in graph.groups:
            group_edges=[]
            for v in group:
                e=len(edges); edges.append(v)
                adjacency[v].append(e); group_edges.append(e)
            # Edge ordering is the factor CSR order.
        messages=unary.new_full((len(edges),2),-0.6931471805599453)
        maximum=self.theta.max()
        theta=self.theta-torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        history=[]
        for _ in range(steps):
            vtof=[None]*len(edges)
            for v,es in enumerate(adjacency):
                excluded,_total=_excluded_sums([messages[e] for e in es],unary.new_zeros(2))
                for e,ex in zip(es,excluded):
                    vtof[e]=normalize(unary[v]+ex)
            vtof=torch.stack(vtof)
            messages=torch.cat([factor_messages(vtof[graph.factor_ptr[f]:graph.factor_ptr[f+1]],theta)
                                for f in range(len(graph.groups))])
            beliefs=normalize(torch.stack([unary[v]+sum((messages[e] for e in es),unary.new_zeros(2))
                                          for v,es in enumerate(adjacency)]))
            history.append(beliefs)
        if not history:
            beliefs=normalize(unary)
        return (beliefs,history) if return_history else beliefs
