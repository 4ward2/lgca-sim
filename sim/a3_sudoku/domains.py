"""Monotone GAC over masks 0=empty, 1=false, 2=true, 3=unknown."""
from .oracle import check_puzzle, validate_solution

EO_RELATION=(False,True,False)


def init_domains(graph,givens):
    check_puzzle(givens,graph.n,graph.regions)
    return [3 if v==-1 else (2 if a==v else 1) for v in givens for a in range(graph.n)]


def _merge(a,b):
    return tuple(any(a[i] and b[j] for i in range(3) for j in range(3) if min(2,i+j)==k)
                 for k in range(3))


def factor_support(domains,relation=EO_RELATION):
    """Exclude-self prefix/suffix convolution, including empty-domain inputs."""
    if len(relation)!=3 or any(type(d) is not int or d not in range(4) for d in domains):
        raise ValueError('invalid domains/relation')
    leaves=[(bool(d&1),bool(d&2),False) for d in domains]
    identity=(True,False,False)
    prefix=[identity]
    for leaf in leaves:
        prefix.append(_merge(prefix[-1],leaf))
    suffix=[identity]*(len(leaves)+1)
    for i in range(len(leaves)-1,-1,-1):
        suffix[i]=_merge(leaves[i],suffix[i+1])
    out=[]
    for i in range(len(leaves)):
        other=_merge(prefix[i],suffix[i+1])
        out.append(sum((1<<h) for h in (0,1)
                       if any(other[k] and relation[min(2,k+h)] for k in range(3))))
    return out


def synchronous_closure(graph,domains,unary=None,relation=EO_RELATION,return_history=False):
    if len(domains)!=graph.num_vars:
        raise ValueError('wrong domain length')
    unary=[3]*graph.num_vars if unary is None else list(unary)
    if len(unary)!=graph.num_vars or any(type(x) is not int or x not in range(4) for x in list(domains)+unary):
        raise ValueError('invalid unary/domains')
    d=[a & b for a,b in zip(domains,unary)]
    history=[d[:]]
    while True:
        nxt=d[:]
        for group in graph.groups:
            support=factor_support([d[v] for v in group],relation)
            for v,s in zip(group,support):
                nxt[v]&=s
        history.append(nxt[:])
        if nxt==d:
            return (nxt,history) if return_history else nxt
        d=nxt


def candidate_sets(graph,domains):
    return [{v for v in range(graph.n) if domains[c*graph.n+v]&2} for c in range(len(graph.cells))]


def read_solution(graph,domains,givens=None):
    """An answer requires all Boolean variables fixed plus independent validation."""
    if len(domains)!=graph.num_vars or any(d not in (1,2) for d in domains):
        return None
    candidates=candidate_sets(graph,domains)
    if any(len(s)!=1 for s in candidates):
        return None
    solution=[next(iter(s)) for s in candidates]
    return solution if validate_solution(solution,graph.n,givens,graph.regions) else None
