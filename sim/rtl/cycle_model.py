"""Low-memory cycle fallback: scalar LUT muxes and registered synchronous state.
Not an HDL simulator; actual XSim coverage is reported separately.
"""
import itertools,json,time
from pathlib import Path
import numpy as np
from sim.cert import tables
from sim.rtl.reference_trace import trace
from sim.rtl.test_cosim import oracle
from sim.rtl.gen_sv import checked,generate

class CycleCore:
    def __init__(self,t,shape):
        self.t=t; self.shape=shape; self.n=int(np.prod(shape)); self.reset()
    def reset(self):
        self.codes=[]; self.dom=[3]*self.n; self.busy=False; self.done=False; self.status=0; self.rounds=0; self.total=0; self.emitted=0; self.computes=0
    @property
    def ready(self): return not self.busy and len(self.codes)<self.n
    @property
    def valid(self): return self.done and self.status==1 and self.emitted<self.n
    @property
    def bit(self): return self.t['o'][self.codes[self.emitted]][self.dom[self.emitted]] if self.valid else 0
    def tick(self,rst=False,start=False,in_valid=False,code=0,out_ready=False):
        if rst: self.reset(); return
        was_ready=self.ready; was_valid=self.valid
        # Pre-edge load count, matching nonblocking assignments.
        count=len(self.codes)
        if in_valid and was_ready: self.codes.append(code)
        if start and not self.busy and count==self.n and not was_valid:
            self.busy=True; self.done=False; self.status=0; self.rounds=0; self.computes=0; self.emitted=0; self.total=1; self.dom=[3]*self.n
        elif self.busy:
            nxt=[]
            for i,p in enumerate(np.ndindex(self.shape)):
                x=self.codes[i]; bits=list(self.t['u'][x])
                for e in range(len(self.shape)):
                    for delta in (-1,1):
                        q=list(p); q[e]+=delta
                        if 0<=q[e]<self.shape[e]:
                            k=int(np.ravel_multi_index(tuple(q),self.shape)); y=self.codes[k]; d=self.dom[k]
                        else: y=0; d=3
                        a,b=(y,x) if delta<0 else (x,y)
                        for h in range(2):
                            support=0
                            for j in range(2):
                                relation=self.t['r'][e][a][b][j][h] if delta<0 else self.t['r'][e][a][b][h][j]
                                support |= ((d>>j)&1)&relation
                            bits[h] &= support
                nxt.append(bits[0]|bits[1]<<1)
            self.computes+=1; self.total+=1
            if nxt==self.dom: self.busy=False; self.done=True; self.status=1
            elif self.computes>=2*self.n+1: self.busy=False; self.done=True; self.status=2
            else: self.rounds+=1
            self.dom=nxt
        elif was_valid:
            self.total+=1
            if out_ready:
                self.emitted+=1
                if self.emitted==self.n: self.codes=[]

def exercise(t,shape,cases):
    core=CycleCore(t,shape); comparisons=0; metrics=[]
    for xi in cases:
        states,out,r=trace(t,xi)
        core.tick(start=True); assert not core.busy # premature start
        for x in xi.flat:
            before=len(core.codes); core.tick(); assert len(core.codes)==before
            assert core.ready; core.tick(in_valid=True,code=int(x))
        core.tick(start=True); assert core.busy and not core.done and core.total==1
        for j,expected in enumerate(states):
            if j: core.tick(start=True,in_valid=True,code=0)
            assert core.dom==expected.ravel().tolist()
            assert core.rounds==min(j,r) and core.total==j+1
            assert core.done==(j==len(states)-1)
            comparisons+=core.n
        assert core.computes==r+1 and core.status==1
        for bit in out.flat:
            assert core.valid and core.bit==int(bit)
            old=(core.emitted,core.bit,core.dom[:]); core.tick(out_ready=False,start=True)
            assert old==(core.emitted,core.bit,core.dom)
            core.tick(out_ready=True)
        assert not core.valid and core.ready and core.done
        metrics.append(dict(rounds=r,compute=r+1,total_without_stalls=2*core.n+r+2))
    core.tick(rst=True,start=True,in_valid=True); assert not core.done and not core.busy and core.ready
    return comparisons,metrics

def main():
    root=Path(__file__).resolve().parents[2]; rng=np.random.default_rng(110801); results=[]; begin=time.time()
    for path in sorted((root/'sim/cert/fixtures').glob('*_tables.json')):
        t=json.loads(path.read_text()); task=path.name[0]; checked(t,task)
        shapes=[(1,),(2,),(3,),(4,),(16,),(64,)] if task!='M' else [(1,1),(2,2),(3,3),(8,8)]
        for shape in shapes:
            cases=[rng.integers(0,1<<(t['c']+1),shape) for _ in range(8)]
            cases += [np.full(shape,x,dtype=int) for x in range(1<<(t['c']+1))]
            if task!='M' and shape[0]<=4:
                cases += [np.array(x).reshape(shape) for x in itertools.product(range(1<<(t['c']+1)),repeat=shape[0])]
            comparisons,metrics=exercise(t,shape,cases)
            # Frozen simulator confirmation at each size, independent of scalar model.
            tables.confirm(t,cases[0])
            generate(t,task,list(shape),root/'sim/rtl/build/generated'/f'{path.stem}_{"x".join(map(str,shape))}',__import__('hashlib').sha256(path.read_bytes()).hexdigest())
            results.append(dict(table=path.name,shape=shape,cases=len(cases),comparisons=comparisons,rounds=[m['rounds'] for m in metrics]))
        print(path.name,flush=True)
    for task in ('P','A','M'):
        t=oracle(task); shape=(4,) if task!='M' else (2,2)
        cases=[np.array(x).reshape(shape) for x in itertools.product(range(4),repeat=4)]
        count,metrics=exercise(t,shape,cases); results.append(dict(table='oracle_'+task,shape=shape,cases=len(cases),comparisons=count,rounds=[m['rounds'] for m in metrics]))
    output=dict(backend='pure Python scalar cycle model (not HDL simulation)',seed=110801,tables=30,designs=len(results),cases=sum(r['cases'] for r in results),domain_comparisons=sum(r['comparisons'] for r in results),seconds=time.time()-begin,results=results)
    (root/'sim/rtl/build/model_results.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='results'}))
if __name__=='__main__': main()
