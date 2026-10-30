"""Fixed strict rounding and size-independent EO semantic certification."""
import math


def cardinality_to_relation(theta,thr=2.):
    values=[float(x) for x in theta]
    if len(values)!=3 or not all(math.isfinite(x) for x in values) or not math.isfinite(thr) or thr<=0:
        raise ValueError('finite three-slot potential and positive threshold required')
    maximum=max(values)
    return tuple(x-maximum > -thr for x in values)


def certify_eo(relation):
    """Three count classes cover every arity; only count one may be accepted."""
    return len(relation)==3 and tuple(relation)==(False,True,False)
