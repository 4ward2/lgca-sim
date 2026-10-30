"""Immutable physical-cell EO hypergraphs with reciprocal CSR incidence."""
from dataclasses import dataclass
from .oracle import standard_regions


def canonical_factor_key(members):
    return tuple(sorted(members))


@dataclass(frozen=True)
class Graph:
    n: int
    cells: tuple
    regions: tuple
    groups: tuple
    factor_type: tuple
    factor_ptr: tuple
    factor_vars: tuple
    var_ptr: tuple
    var_factors: tuple
    boards: tuple

    @property
    def num_vars(self):
        return len(self.cells)*self.n


def _make(n,cells,regions,boards):
    groups=[tuple(c*n+v for v in range(n)) for c in range(len(cells))]
    types=['CELL']*len(cells)
    seen=set(map(canonical_factor_key,groups))
    for region,kind in regions:
        for v in range(n):
            group=tuple(c*n+v for c in region)
            key=canonical_factor_key(group)
            if key not in seen:
                groups.append(group)
                types.append(kind)
                seen.add(key)
    fp=[0]
    fv=[]
    adjacency=[[] for _ in range(len(cells)*n)]
    for f,group in enumerate(groups):
        fv.extend(group)
        fp.append(len(fv))
        for v in group:
            adjacency[v].append(f)
    vp=[0]
    vf=[]
    for fs in adjacency:
        vf.extend(fs)
        vp.append(len(vf))
    g=Graph(n,tuple(cells),tuple(r for r,_ in regions),tuple(groups),tuple(types),
            tuple(fp),tuple(fv),tuple(vp),tuple(vf),tuple(boards))
    validate_graph(g)
    return g


def build_standard(n):
    rs=standard_regions(n)
    return _make(n,[(r,c) for r in range(n) for c in range(n)],
                 [(r,('ROW_VALUE','COL_VALUE','BOX_VALUE')[i//n]) for i,r in enumerate(rs)],
                 [tuple(range(n*n))])


def build_samurai():
    starts=((0,0),(0,12),(6,6),(12,0),(12,12))
    cells=sorted({(r+i,c+j) for r,c in starts for i in range(9) for j in range(9)})
    ids={xy:i for i,xy in enumerate(cells)}
    boards=[tuple(ids[r+i,c+j] for i in range(9) for j in range(9)) for r,c in starts]
    rs=[]
    for board in boards:
        rs.extend((tuple(board[c] for c in reg),('ROW_VALUE','COL_VALUE','BOX_VALUE')[i//9])
                  for i,reg in enumerate(standard_regions(9)))
    return _make(9,cells,rs,boards)


def validate_graph(g):
    def require(condition):
        if not condition:
            raise ValueError('malformed Sudoku graph')
    require(g.n>=1 and len(set(g.cells))==len(g.cells))
    require(len(g.factor_type)==len(g.groups))
    require(len(g.factor_ptr)==len(g.groups)+1 and g.factor_ptr[0]==0)
    require(g.factor_ptr[-1]==len(g.factor_vars))
    require(len(g.var_ptr)==g.num_vars+1 and g.var_ptr[0]==0)
    require(g.var_ptr[-1]==len(g.var_factors))
    require(all(a<=b for a,b in zip(g.factor_ptr,g.factor_ptr[1:])))
    require(all(a<=b for a,b in zip(g.var_ptr,g.var_ptr[1:])))
    require(len(set(map(canonical_factor_key,g.groups)))==len(g.groups))
    expected=[[] for _ in range(g.num_vars)]
    for f,group in enumerate(g.groups):
        require(len(group)==g.n and len(set(group))==g.n)
        require(tuple(g.factor_vars[g.factor_ptr[f]:g.factor_ptr[f+1]])==group)
        require(g.factor_type[f] in ('CELL','ROW_VALUE','COL_VALUE','BOX_VALUE'))
        for v in group:
            require(type(v) is int and 0<=v<g.num_vars)
            expected[v].append(f)
    for v,fs in enumerate(expected):
        require(bool(fs) and tuple(fs)==g.var_factors[g.var_ptr[v]:g.var_ptr[v+1]])
    require(all(len(r)==g.n and len(set(r))==g.n and all(0<=c<len(g.cells) for c in r)
                for r in g.regions))
    cellgroups={tuple(c*g.n+v for v in range(g.n)) for c in range(len(g.cells))}
    require({g.groups[f] for f,t in enumerate(g.factor_type) if t=='CELL'}==cellgroups)
    regiongroups={canonical_factor_key(c*g.n+v for c in r) for r in g.regions for v in range(g.n)}
    require({canonical_factor_key(gr) for gr,t in zip(g.groups,g.factor_type) if t!='CELL'}==regiongroups)
    return True
