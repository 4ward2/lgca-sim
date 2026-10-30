"""Complete EO DFS. Node = a visited domain state, including root and conflicts."""
import argparse
import json
import time
from pathlib import Path
from .domains import init_domains,synchronous_closure,candidate_sets,read_solution
from .graph import build_standard


def undo_trail(domains,trail,mark):
    while len(trail)>mark:
        v,old=trail.pop(); domains[v]=old


def choose_mrv(graph,candidates):
    peers=[set().union(*(set(r) for r in graph.regions if c in r))-{c} for c in range(len(graph.cells))]
    return min((c for c,s in enumerate(candidates) if len(s)>1),
               key=lambda c:(len(candidates[c]),-sum(len(candidates[j])>1 for j in peers[c]),c))


def order_learned(model,graph,domains,cell,values,steps=16):
    import torch
    from .soft import batched_forward
    # All branch singleton cells are safe hard clamps; other deletions remain
    # exclusively in the complete EO solver. Scores only permute candidates.
    candidates=candidate_sets(graph,domains)
    givens=[next(iter(s)) if len(s)==1 else -1 for s in candidates]
    with torch.no_grad():
        p=batched_forward(model,graph,[givens],steps)[0,:,1].exp().reshape(-1,graph.n)[cell]
    return sorted(values,key=lambda v:(-float(p[v]),v))


def solve(puzzle,n,model=None,node_limit=100000,time_limit=60,steps=16,trace=False):
    if node_limit<0 or time_limit<0:
        raise ValueError('nonnegative budgets required')
    graph=build_standard(n); d=init_domains(graph,puzzle); trail=[]
    start=time.monotonic(); deadline=start+time_limit
    stats=dict(nodes=0,decisions=0,conflicts=0,propagation_seconds=0.,ordering_seconds=0.)
    visits=[]
    def write(new):
        for v,(old,value) in enumerate(zip(d,new)):
            if old!=value:
                trail.append((v,old)); d[v]=value
    def dfs():
        if stats['nodes']>=node_limit or time.monotonic()>=deadline:
            raise TimeoutError
        stats['nodes']+=1
        t=time.monotonic(); write(synchronous_closure(graph,d)); stats['propagation_seconds']+=time.monotonic()-t
        if time.monotonic()>=deadline:
            raise TimeoutError
        if 0 in d:
            stats['conflicts']+=1; return None
        answer=read_solution(graph,d,puzzle)
        if answer is not None:
            return answer
        candidates=candidate_sets(graph,d); c=choose_mrv(graph,candidates)
        t=time.monotonic()
        values=sorted(candidates[c]) if model is None else order_learned(model,graph,d,c,candidates[c],steps)
        stats['ordering_seconds']+=time.monotonic()-t
        if sorted(values)!=sorted(candidates[c]):
            raise ValueError('ordering must preserve candidates')
        for value in values:
            if trace:
                visits.append([c,value])
            stats['decisions']+=1; mark=len(trail)
            for v in range(n):
                i=c*n+v; trail.append((i,d[i])); d[i]=2 if v==value else 1
            try:
                answer=dfs()
            finally:
                undo_trail(d,trail,mark)
            if answer is not None:
                return answer
        return None
    try:
        answer=dfs(); status='SAT' if answer is not None else 'UNSAT'
    except TimeoutError:
        answer=None; status='BUDGET'
    result=dict(status=status,solution=answer,seconds=time.monotonic()-start,**stats)
    if trace:
        result['trace']=visits
    return result


def main():
    from .train import load_records
    from .soft import CardinalityBP
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',nargs='+',required=True); p.add_argument('--output',required=True)
    p.add_argument('--checkpoint'); p.add_argument('--node-limit',type=int,default=100000)
    p.add_argument('--time-limit',type=float,default=60); a=p.parse_args()
    model=None
    if a.checkpoint:
        model=CardinalityBP(); model.load_state_dict(torch.load(a.checkpoint,weights_only=True,map_location='cpu'))
    results=[dict(sample_id=r['sample_id'],**solve(r['puzzle'],r['n'],model,a.node_limit,a.time_limit)) for r in load_records(a.input,'dev')]
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'search.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
