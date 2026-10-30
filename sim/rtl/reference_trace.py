"""Full synchronous trace, including the final equal comparison."""
import itertools
import numpy as np
from sim.cert import tables

def trace(t, xi):
    xi=tables.validate_input(t,xi); old=np.full(xi.shape,3,dtype=int); states=[old.copy()]
    for _ in range(2*xi.size+1):
        nxt=np.empty_like(old)
        for p in np.ndindex(xi.shape):
            x=int(xi[p]); s=tables.unary(t,x)
            for axis,delta in itertools.product(range(xi.ndim),(-1,1)):
                q=list(p); q[axis]+=delta; q=tuple(q)
                inside=all(0<=q[k]<xi.shape[k] for k in range(xi.ndim))
                y,d=(int(xi[q]),int(old[q])) if inside else (0,3)
                s &= tables.support(t,axis,y,x,d,True) if delta<0 else tables.support(t,axis,x,y,d,False)
            nxt[p]=s
        states.append(nxt.copy())
        if np.array_equal(old,nxt): break
        old=nxt
    else: raise AssertionError('timeout')
    final,out,r=tables.interpret(t,xi)
    assert np.array_equal(final,states[-1]) and r==len(states)-2
    return states,out,r
