"""Materialize a reviewed arm matrix without launching experiments."""
import copy
from .shards import validate_dag

def materialize(plan,freeze):
    if freeze.get('gate',{}).get('status')!='READY_FOR_REVIEW':
        raise ValueError('pilot gate has not passed')
    enabled=set(freeze['gate']['arms'])
    result=copy.deepcopy(plan)
    def keep(job):
        for arm in ('b','r','f','dt-small','shared64','unshared64'):
            if job['name'].startswith((f'train-{arm}-',f'eval-{arm}-')):
                return arm in enabled
        return True
    result['jobs']=[j for j in result['jobs'] if keep(j)]
    names={j['name'] for j in result['jobs']}
    for job in result['jobs']: job['needs']=[n for n in job['needs'] if n in names]
    result['status']='DRAFT-REVIEWED-MATRIX'
    result['arms']=sorted(enabled)
    validate_dag(result)
    return result
