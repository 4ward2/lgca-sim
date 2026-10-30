"""V=2-only all-length wrapper with bounded worker and independent replay."""
import multiprocessing as mp
import time
import numpy as np
from sim.cert import chain,tables
from .protocol import digest

def scalar_replay(table,word):
    # Separate scalar loop, keeping virtual domains full and ignoring u[0].
    domains=[3]*len(word)
    for step in range(2*len(word)+1):
        nxt=[]
        for i,x in enumerate(word):
            value=0
            for h in range(2):
                ok=bool(table['u'][x][h])
                for delta in (-1,1):
                    j=i+delta; inside=0<=j<len(word)
                    z=word[j] if inside else 0; domain=domains[j] if inside else 3
                    ok &= any(domain&(1<<k) and (table['r'][0][z][x][k][h] if delta<0 else table['r'][0][x][z][h][k]) for k in range(2))
                value|=int(ok)<<h
            nxt.append(value)
        if nxt==domains: return [table['o'][x][d] for x,d in zip(word,domains)]
        domains=nxt
    raise AssertionError('scalar convergence')

def _worker(table,connection):
    try: connection.send(('ok',chain.decide(table,'P')))
    except Exception as e: connection.send(('error',repr(e)))
    finally: connection.close()

def certify_prefix(table,timeout=60):
    tables.validate(table,'P')
    if table['V']!=2: raise ValueError('A4 rejects V != 2')
    actual=digest({k:table[k] for k in ('dims','c','V','u','r','o')})
    # tables.hashable uses identical sorted JSON serialization.
    if table.get('table_sha256') not in (None,actual): raise ValueError('wrong table SHA')
    start=time.perf_counter(); ctx=mp.get_context('spawn'); parent,child=ctx.Pipe(False)
    process=ctx.Process(target=_worker,args=(table,child)); process.start(); child.close()
    if not parent.poll(timeout):
        process.terminate(); process.join(); result=dict(verdict='UNKNOWN',reason='timeout')
    else:
        try:
            status,result=parent.recv()
            if status!='ok': result=dict(verdict='UNKNOWN',reason=result)
        except EOFError:
            result=dict(verdict='UNKNOWN',reason='checker worker exited without result')
        finally:
            process.join()
    parent.close()
    result.update(seconds=time.perf_counter()-start,table_sha256=actual,**tables.provenance())
    if result['verdict']=='FAIL':
        word=result['counterexample']; pred=scalar_replay(table,word)
        if pred==result['expected']: raise AssertionError('counterexample failed independent replay')
        result['scalar_prediction']=pred
    return result

def replay(table,payload):
    # Recompute complete automata; do not trust payload verdict or omitted obligations.
    current=certify_prefix(table)
    if payload.get('checker_sha256')!=current['checker_sha256']:
        raise ValueError('checker source identity mismatch')
    for field in ('verdict','table_sha256','left_states','right_states','counterexample','expected'):
        if payload.get(field)!=current.get(field): raise ValueError('proof mismatch: '+field)
    return current
