"""Offline random DFS solutions, uniqueness-preserving removal and fixed family split."""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from .data import _seed, symmetry_key, classify_level
from .oracle import standard_regions, count_solutions
from .manifest import digest, make_manifest


def random_solution(n,rng,deadline):
    regions=standard_regions(n)
    peers=[set().union(*(set(r) for r in regions if c in r))-{c} for c in range(n*n)]
    board=[-1]*(n*n)
    def visit():
        if time.monotonic()>=deadline:
            raise TimeoutError
        options=[(set(range(n))-{board[j] for j in peers[c]},c) for c in range(n*n) if board[c]<0]
        if not options:
            return True
        rng.shuffle(options)
        values,c=min(options,key=lambda x:len(x[0]))
        values=list(sorted(values)); rng.shuffle(values)
        for v in values:
            board[c]=v
            if visit():
                return True
        board[c]=-1
        return False
    if not visit():
        raise RuntimeError('empty standard board has no completion')
    return board


def generate(n,count,seed,time_limit,split,resume=None):
    _seed(seed)
    if n not in (4,9) or count<1 or time_limit<0 or split not in ('train','dev'):
        raise ValueError('invalid generation budget/shape/split')
    rng=random.Random(seed); deadline=time.monotonic()+time_limit
    records=[]; seen=set(); attempts=0
    if resume:
        if [resume[k] for k in ('n','seed','split')]!=[n,seed,split]:
            raise ValueError('generation resume identity mismatch')
        records=resume['records']; attempts=resume['attempts']
        seen={symmetry_key(r['puzzle'],n) for r in records}
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value,list) else value
        rng.setstate(tuples(resume['rng_state']))
    try:
        while len(records)<count:
            solution=random_solution(n,rng,deadline); attempts+=1
            # Fixed hash independent of generation seed: same checked solution family
            # always belongs to the same split, even across separate artifact jobs.
            family=digest([n,symmetry_key(solution,n)])
            assigned='dev' if int(digest([6,family]),16)%5==0 else 'train'
            if assigned!=split:
                continue
            puzzle=solution[:]; cells=list(range(n*n)); rng.shuffle(cells)
            for c in cells:
                old=puzzle[c]; puzzle[c]=-1
                if count_solutions(puzzle,n,limit=2,timeout=max(0,deadline-time.monotonic()))!=1:
                    puzzle[c]=old
            key=symmetry_key(puzzle,n)
            if key in seen:
                continue
            label=classify_level(puzzle,n,max(0,deadline-time.monotonic()))
            if label['uniqueness']!='UNIQUE':
                continue
            if time.monotonic()>=deadline:
                raise TimeoutError
            seen.add(key)
            records.append(dict(n=n,puzzle=puzzle,solution=solution,source='random-dfs',row_id=str(attempts),
                                family=family,split=split,generation_seed=seed,**label))
    except TimeoutError:
        pass
    return dict(n=n,seed=seed,split=split,rng_state=rng.getstate(),records=records,requested=count,attempts=attempts,status='complete' if len(records)==count else 'budget',
                counts=dict(Counter(r['level'] for r in records)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,required=True); p.add_argument('--count',type=int,required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--time-limit',type=float,required=True)
    p.add_argument('--split',choices=['train','dev'],required=True); p.add_argument('--output',required=True)
    p.add_argument('--resume',help='previous output checkpoint.json')
    a=p.parse_args()
    resume=json.loads(Path(a.resume).read_text(encoding='utf-8-sig')) if a.resume else None
    result=generate(a.n,a.count,a.seed,a.time_limit,a.split,resume)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoint.json').write_text(json.dumps(result),encoding='utf-8')
    splits={s:result['records'] if s==a.split else [] for s in ('train','dev','test')}
    manifest=make_manifest(splits,a.seed,provenance=dict(generator='random-dfs',**{k:v for k,v in result.items() if k not in ('records','rng_state')}))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'puzzles.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest['records']),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('records','rng_state')},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','rng_state')}))

if __name__=='__main__':
    main()
