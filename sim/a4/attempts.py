"""Resolve explicit infrastructure retries without hiding algorithm failures."""
from .protocol import digest

def resolve_attempts(attempts,max_infrastructure_retries=2):
    if not attempts: raise ValueError('missing attempt chain')
    if any(not isinstance(a,dict) for a in attempts): raise ValueError('invalid attempt record')
    ordered=sorted(attempts,key=lambda a:a.get('attempt',-1))
    if [a.get('attempt') for a in ordered]!=list(range(len(ordered))):
        raise ValueError('duplicate or discontinuous attempt chain')
    identities={digest(a.get('identity')) for a in ordered}
    if len(identities)!=1 or ordered[0].get('identity') is None: raise ValueError('retry identity changed')
    if len(ordered)>max_infrastructure_retries+1: raise ValueError('infrastructure retry allowance exceeded')
    for previous,current in zip(ordered,ordered[1:]):
        if previous.get('status')!='infrastructure_failure' or not previous.get('reason'):
            raise ValueError('only documented infrastructure failures may retry')
        if current.get('parent_attempt_sha256')!=digest(previous): raise ValueError('broken attempt provenance')
    final=ordered[-1]
    allowed={'complete','infrastructure_failure','nonfinite_loss','budget_exceeded','algorithm_failure'}
    if final.get('status') not in allowed: raise ValueError('unknown attempt status')
    return final
