"""Offline pair validation, conservative family splitting and augmentation.

Supported text schemas: Kaggle 1M quizzes/solutions; Kaggle 3M puzzle/solution
(metadata/rating retained); Extreme question/answer (source/rating retained).
Strings use 1..9, A..P for 16/25; 0 and . are blanks. Lists are zero-based,
with -1 blanks. No network I/O. Confirm public revisions/column names on ingest.
"""
import csv
import hashlib
import itertools
import json
import random
from math import isqrt
from pathlib import Path
from .oracle import check_puzzle, validate_solution, count_solutions, singles_closure, hall_closure

SYMBOLS='123456789ABCDEFGHIJKLMNOP'


def _seed(seed):
    if type(seed) is not int or seed<60000:
        raise ValueError('development seed must be >=60000')


def _decode(value,n):
    if isinstance(value,str):
        chars=''.join(value.split()).upper()
        out=tuple(-1 if c in '.0' else SYMBOLS.index(c) for c in chars)
    else:
        out=tuple(value)
    if len(out)!=n*n or any(type(x) is not int or x < -1 or x>=n for x in out):
        raise ValueError('invalid puzzle encoding')
    return out


def parse_pairs(rows,n=9,source='local',uniqueness_seconds=.25,max_records=1000):
    """Validate each pair and independently label UNIQUE/MULTI/UNVERIFIED.

    A path may be CSV or JSONL. Iterables of mappings are also supported.
    max_records bounds consumed input; uniqueness_seconds bounds each search.
    """
    if max_records<1 or uniqueness_seconds<0:
        raise ValueError('explicit positive record budget and nonnegative timeout required')
    if isinstance(rows,(str,Path)):
        with open(rows,encoding='utf-8-sig',newline='') as f:
            iterator=(json.loads(line) for line in f if line.strip()) if str(rows).endswith('.jsonl') else csv.DictReader(f)
            return parse_pairs(iterator,n,source,uniqueness_seconds,max_records)
    records=[]
    for i,row in enumerate(itertools.islice(rows,max_records)):
        p=next((row[k] for k in ('quizzes','puzzle','question') if k in row),None)
        s=next((row[k] for k in ('solutions','solution','answer') if k in row),None)
        if p is None or s is None:
            raise ValueError('missing puzzle/solution columns')
        puzzle,solution=_decode(p,n),_decode(s,n)
        check_puzzle(puzzle,n)
        if not validate_solution(solution,n,puzzle):
            raise ValueError(f'invalid supplied solution at row {i}')
        try:
            count=count_solutions(puzzle,n,timeout=uniqueness_seconds)
            unique='UNIQUE' if count==1 else ('MULTI' if count>1 else 'UNSAT')
        except TimeoutError:
            unique='UNVERIFIED'
        records.append(dict(n=n,puzzle=puzzle,solution=solution,source=source,row_id=str(row.get('id',i)),
                            family=str(row.get('family',f'{source}:{i}')),uniqueness=unique,
                            metadata=dict(row)))
    return records


def _renumber(sequence):
    names={}
    return tuple(-1 if v==-1 else names.setdefault(v,len(names)) for v in sequence)


def symmetry_key(puzzle,n):
    """Exact 4x4 canonicalization; checked dihedral+digit orbit for larger N.

    Not a full 9/16/25 isomorphism test. Larger-board row/band equivalents may
    remain undetected and this limitation is recorded in every manifest.
    """
    b=isqrt(n)
    if n==4:
        axes=[]
        for bands in itertools.permutations(range(b)):
            for within in itertools.product(list(itertools.permutations(range(b))),repeat=b):
                axes.append(tuple(block*b+i for block,perm in zip(bands,within) for i in perm))
    else:
        axes=[tuple(range(n)),tuple(reversed(range(n)))]
    candidates=[]
    for rows in axes:
        for cols in axes:
            for transpose in (False,True):
                candidates.append(_renumber([puzzle[cols[c]*n+rows[r] if transpose else rows[r]*n+cols[c]]
                                              for r in range(n) for c in range(n)]))
    return min(candidates)


def family_split(records,seed=60000):
    """Union families BEFORE assignment; remove cross-source checked duplicates.

    Same supplied full solution (up to checked symmetries) is conservatively
    one family even when the upstream family metadata is absent.
    """
    _seed(seed)
    records=[dict(r) for r in records]
    parent=list(range(len(records)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        parent[root(i)]=root(j)
    seen={}; puzzle_keys=[]
    for i,r in enumerate(records):
        pk=(r['n'],symmetry_key(r['puzzle'],r['n']))
        puzzle_keys.append(pk)
        keys=[('family',r['family']),('puzzle',pk),('solution',r['n'],symmetry_key(r['solution'],r['n']))]
        for key in keys:
            if key in seen:
                union(i,seen[key])
            else:
                seen[key]=i
    families={}
    origins={}
    for i,r in enumerate(records):
        families.setdefault(root(i),set()).add(r['family'])
        origins.setdefault(puzzle_keys[i],[]).append(dict(source=r['source'],row_id=r['row_id']))
    result={k:[] for k in ('train','dev','test')}
    used=set()
    for i,r in enumerate(records):
        if puzzle_keys[i] in used:
            continue
        used.add(puzzle_keys[i])
        fs=sorted(families[root(i)])
        number=int(hashlib.sha256(json.dumps([seed,fs]).encode()).hexdigest(),16)%100
        split='train' if number<80 else ('dev' if number<90 else 'test')
        r.update(split=split,families=fs,origins=origins[puzzle_keys[i]])
        result[split].append(r)
    return result


def augment(record,seed=60000):
    _seed(seed)
    if record.get('split') not in ('train','dev','test'):
        raise ValueError('split original family before augmentation')
    rng=random.Random(seed); n=record['n']; b=isqrt(n)
    def axis():
        bands=list(range(b)); rng.shuffle(bands)
        out=[]
        for band in bands:
            within=list(range(b)); rng.shuffle(within)
            out.extend(band*b+i for i in within)
        return out
    rows,cols=axis(),axis(); digits=list(range(n)); rng.shuffle(digits)
    transpose=rng.choice((False,True))
    out=dict(record)
    for key in ('puzzle','solution'):
        board=record[key]
        out[key]=tuple(-1 if (v:=board[cols[c]*n+rows[r] if transpose else rows[r]*n+cols[c]])==-1 else digits[v]
                       for r in range(n) for c in range(n))
    out['augmentation_seed']=seed
    return out


def classify_level(puzzle,n,uniqueness_seconds=.25):
    try:
        check_puzzle(puzzle,n)
    except ValueError:
        return dict(level='INVALID',uniqueness='INVALID')
    try:
        count=count_solutions(puzzle,n,timeout=uniqueness_seconds)
    except TimeoutError:
        return dict(level='UNVERIFIED',uniqueness='UNVERIFIED')
    if count!=1:
        return dict(level='UNSAT' if count==0 else 'MULTI',uniqueness='UNSAT' if count==0 else 'MULTI')
    singles,info=singles_closure(puzzle,n,return_info=True)
    level=('S0' if all(v>=0 for v in puzzle) else 'S1' if all(len(s)==1 for s in singles)
           else 'S2' if all(len(s)==1 for s in hall_closure(puzzle,n)) else 'S3')
    return dict(level=level,uniqueness='UNIQUE',remaining_candidates=info['remaining_candidates'],singles_rounds=info['rounds'])
