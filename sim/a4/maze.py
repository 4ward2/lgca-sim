"""Optional format/feasibility gate; no maze training."""
from collections import deque
import numpy as np
from .data import decode_maze,encode_maze_mask

DIRECTIONS=((-1,0),(0,1),(1,0),(0,-1)) # N,E,S,W

def bfs_oracle(open_cells,source,goal):
    start=tuple(np.argwhere(source)[0]); end=tuple(np.argwhere(goal)[0]); parents={start:None}; queue=deque([start])
    edges=0
    while queue:
        p=queue.popleft()
        for dy,dx in DIRECTIONS:
            q=(p[0]+dy,p[1]+dx)
            if 0<=q[0]<open_cells.shape[0] and 0<=q[1]<open_cells.shape[1] and open_cells[q]:
                edges+=1
                if q not in parents: parents[q]=p; queue.append(q)
    if len(parents)!=int(open_cells.sum()) or edges//2!=len(parents)-1 or end not in parents:
        raise ValueError('maze is not a connected tree')
    mask=np.zeros_like(open_cells)
    p=end
    while p is not None: mask[p]=1; p=parents[p]
    return mask

def parent_path(parents,source,goal,open_cells):
    p=tuple(np.argwhere(goal)[0]); start=tuple(np.argwhere(source)[0]); visited=set(); mask=np.zeros_like(open_cells)
    for _ in range(open_cells.size):
        if p in visited or not (0<=p[0]<mask.shape[0] and 0<=p[1]<mask.shape[1]) or not open_cells[p]: raise ValueError('invalid parent path')
        visited.add(p); mask[p]=1
        if p==start: return mask
        if p not in parents: raise ValueError('goal unreached or missing parent')
        q=parents[p]
        if abs(q[0]-p[0])+abs(q[1]-p[1])!=1: raise ValueError('nonadjacent parent')
        p=q
    raise ValueError('parent traversal limit')

def wave_parent(open_cells,source,goal):
    reached=source.copy(); parents={}
    for _ in range(open_cells.size):
        nxt=reached.copy()
        for p in map(tuple,np.argwhere(open_cells & ~reached)):
            for dy,dx in DIRECTIONS:
                q=(p[0]+dy,p[1]+dx)
                if 0<=q[0]<reached.shape[0] and 0<=q[1]<reached.shape[1] and reached[q]:
                    nxt[p]=1; parents[p]=q; break
        if np.array_equal(nxt,reached): break
        reached=nxt
    return parent_path(parents,source,goal,open_cells)

def maze_gate(rgb,targets):
    for x,y in zip(rgb,targets):
        d=decode_maze(x,y)
        if not np.array_equal(encode_maze_mask(d['rgb']),x): raise ValueError('RGB roundtrip')
        expected=bfs_oracle(d['open'],d['source'],d['goal'])
        if not np.array_equal(expected,d['target']): raise ValueError('public target differs from unique path')
        if not np.array_equal(wave_parent(d['open'],d['source'],d['goal']),expected): raise ValueError('wave mismatch')
    return dict(status='format-feasibility-pass',count=len(rgb),learned_wave_enabled=False)
