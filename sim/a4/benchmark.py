"""Synthetic-only latency measurements. Never reads benchmark OOD examples."""
import time
import copy
import numpy as np
import torch
from sim.logic import to_internal
from .protocol import LENGTHS
from .evaluate import run_ac_to_fp,run_bp,run_dt_streaming
from .models import calibrate,predict

@torch.no_grad()
def synthetic_latency(model,arm,train,warmups=1,repeats=3):
    rows=[]; previous=torch.get_num_threads(); model.eval(); overall_start=time.perf_counter()
    ac=calibrate(model,train) if arm=='b' else None
    qbp=copy.deepcopy(model) if arm=='b' else None
    if qbp is not None: qbp.infer='max'
    engines=['ac','soft','qbp'] if arm=='b' else ['main']
    try:
        for threads in (4,):  # only 4-thread rows enter the budget estimate (pilot fix 2026-10-31)
            torch.set_num_threads(threads)
            for length in LENGTHS:
                for batch in (1,128):
                    x=np.zeros((batch,1,length),np.float32); xv=to_internal(x)
                    for engine in engines:
                        def invoke():
                            if engine=='ac': return run_ac_to_fp(ac,xv)
                            if engine in ('soft','qbp'): return run_bp(qbp if engine=='qbp' else model,xv,2*length+1,hard=engine=='qbp')
                            if arm=='dt-small': return run_dt_streaming(model,torch.tensor(x)-.5,2*length+1)
                            if arm in ('shared64','unshared64'): return predict(model,arm,xv,64)
                            return predict(model,arm,xv,2*length+1,hard=True)
                        for _ in range(warmups): invoke()
                        start=time.perf_counter()
                        for _ in range(repeats): invoke()
                        rows.append(dict(engine=engine,threads=threads,length=length,batch=batch,warmups=warmups,repeats=repeats,seconds_per_batch=(time.perf_counter()-start)/repeats))
                        if time.perf_counter()-overall_start>900:
                            raise ValueError('synthetic timing exceeded fifteen-minute pilot allowance; stop budget qualification')
    finally: torch.set_num_threads(previous)
    hours=sum(r['seconds_per_batch']*int(np.ceil(10000/r['batch']))/3600*(14 if arm=='b' and r['engine']=='ac' else 1) for r in rows if r['threads']==4 and r['batch']==128)
    # Fourteen discrete runs: main, deployment AC, control/flipped, ten random.
    # Continuous engines have their own measured costs above.
    return dict(rows=rows,evaluation_runner_hours=hours,
                budget_qualified=True,
                estimate_basis='synthetic zero inputs; separately measured soft/qBP; fourteen AC evaluations including controls',synthetic=True)
