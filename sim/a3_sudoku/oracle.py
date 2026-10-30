"""Independent cell-set oracles. Values are 0..N-1; blanks are -1."""
from itertools import combinations
from math import isqrt
import time


def standard_regions(n):
    b = isqrt(n)
    if n < 1 or b*b != n:
        raise ValueError('N must be a positive square')
    return ([tuple(r*n+c for c in range(n)) for r in range(n)] +
            [tuple(r*n+c for r in range(n)) for c in range(n)] +
            [tuple((r+i)*n+c+j for i in range(b) for j in range(b))
             for r in range(0,n,b) for c in range(0,n,b)])


def check_puzzle(puzzle,n,regions=None):
    regions = standard_regions(n) if regions is None else regions
    cells = max(max(r) for r in regions)+1
    if len(puzzle)!=cells or any(type(v) is not int or v < -1 or v >= n for v in puzzle):
        raise ValueError('invalid puzzle shape/value')
    for r in regions:
        vals=[puzzle[c] for c in r if puzzle[c]>=0]
        if len(vals)!=len(set(vals)):
            raise ValueError('conflicting givens')
    return regions


def validate_solution(solution,n,givens=None,regions=None):
    try:
        regions=check_puzzle(solution,n,regions)
        return (all(set(solution[c] for c in r)==set(range(n)) for r in regions)
                and (givens is None or (len(givens)==len(solution) and
                     all(v==-1 or solution[c]==v for c,v in enumerate(givens)))))
    except (ValueError,TypeError):
        return False


def count_solutions(puzzle,n,limit=2,timeout=None,regions=None):
    """Independent MRV search; timeout raises, never masquerades as UNSAT."""
    if limit < 1:
        raise ValueError('positive limit required')
    try:
        regions=check_puzzle(puzzle,n,regions)
    except ValueError:
        return 0
    deadline=time.monotonic()+timeout if timeout is not None else float('inf')
    board=list(puzzle)
    peers=[set().union(*(set(r) for r in regions if c in r))-{c} for c in range(len(board))]
    def dfs():
        if time.monotonic()>=deadline:
            raise TimeoutError('uniqueness budget exhausted')
        opts=[(set(range(n))-{board[j] for j in peers[c]},c) for c,v in enumerate(board) if v==-1]
        if not opts:
            return 1
        choices,c=min(opts,key=lambda x:len(x[0]))
        count=0
        for v in sorted(choices):
            board[c]=v
            count+=dfs()
            if count>=limit:
                break
        board[c]=-1
        return min(limit,count)
    return dfs()


def singles_closure(puzzle,n,candidates=None,regions=None,return_info=False):
    regions=standard_regions(n) if regions is None else regions
    if len(puzzle)!=max(max(r) for r in regions)+1 or any(v < -1 or v>=n for v in puzzle):
        raise ValueError('invalid puzzle')
    d=[({v} if v>=0 else set(range(n))) for v in puzzle]
    if candidates is not None:
        if len(candidates)!=len(d) or any(not set(s)<=set(range(n)) for s in candidates):
            raise ValueError('invalid candidate sets')
        d=[a & set(b) for a,b in zip(d,candidates)]
    rounds=0
    def result(domains):
        return (domains,dict(rounds=rounds,remaining_candidates=sum(map(len,domains)))) if return_info else domains
    while True:
        rounds+=1
        nxt=[set(s) for s in d]
        for r in regions:
            for c in r:
                if len(d[c])==1:
                    for j in r:
                        if j!=c:
                            nxt[j]-=d[c]
            for v in range(n):
                places=[c for c in r if v in d[c]]
                if not places:
                    # Canonical contradiction: whole connected Sudoku is inconsistent.
                    return result([set() for _ in d])
                if len(places)==1:
                    nxt[places[0]] &= {v}
        if any(not s for s in nxt):
            return result([set() for _ in d])
        if nxt==d:
            return result(d)
        d=nxt


def hall_closure(puzzle,n,candidates=None,regions=None):
    regions=standard_regions(n) if regions is None else regions
    d=singles_closure(puzzle,n,candidates,regions)
    while all(d):
        nxt=[set(s) for s in d]
        for r in regions:
            for k in (2,3):
                for cells in combinations(r,k):
                    vals=set().union(*(d[c] for c in cells))
                    if len(vals)<k:
                        return [set() for _ in d]
                    if len(vals)==k:
                        for c in set(r)-set(cells):
                            nxt[c]-=vals
                for vals in combinations(range(n),k):
                    cells={c for c in r if d[c]&set(vals)}
                    if len(cells)<k:
                        return [set() for _ in d]
                    if len(cells)==k:
                        for c in cells:
                            nxt[c]&=set(vals)
        nxt=singles_closure(puzzle,n,nxt,regions)
        if nxt==d:
            break
        d=nxt
    return d
