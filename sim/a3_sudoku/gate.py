"""Dev-only go/no-go audit; insufficient S3 coverage never passes the gate."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from .train import load_records,evaluate
from .soft import CardinalityBP
from .rounding import cardinality_to_relation,certify_eo
from .graph import build_standard
from .domains import init_domains,synchronous_closure,candidate_sets
from .oracle import singles_closure
from .search import solve
from .manifest import code_hashes


def continue_decision(results):
    return (len(results)==3 and sum(r['certified'] for r in results)>=2
            and all(r['wrong_deletions']==0 and (r['s1_agreement'] if r['certified'] else True) for r in results)
            and any(r['search_complete'] and r['node_ratio'] is not None and r['node_ratio']<=.8 and r['net_faster'] for r in results))


@torch.no_grad()
def audit(model,records,seed,search_count=100,node_limit=100000,seconds=15,steps=16):
    relation=cardinality_to_relation(model.theta); certified=certify_eo(relation)
    wrong=0; mismatches=0; s1_count=0
    for r in records:
        g=build_standard(r['n']); d=synchronous_closure(g,init_domains(g,r['puzzle']),relation=relation)
        candidates=candidate_sets(g,d)
        # A true Boolean assignment has both chosen=true and all others=false.
        wrong+=sum(not (d[c*g.n+v] & (2 if v==answer else 1)) for c,answer in enumerate(r['solution']) for v in range(g.n))
        if r.get('level')=='S1':
            s1_count+=1
            mismatches+=int(candidates!=singles_closure(r['puzzle'],g.n))
    selected=[r for r in records if r.get('level')=='S3'][:search_count]
    pairs=[]
    for r in selected:
        base=solve(r['puzzle'],r['n'],node_limit=node_limit,time_limit=seconds)
        learned=solve(r['puzzle'],r['n'],model,node_limit,seconds,steps)
        pairs.append(dict(sample_id=r['sample_id'],baseline=base,learned=learned))
    complete=len(pairs)==search_count and all(p['baseline']['status']=='SAT' and p['learned']['status']=='SAT' for p in pairs)
    ratios=[p['learned']['nodes']/p['baseline']['nodes'] for p in pairs if p['baseline']['nodes']]
    btime=sum(p['baseline']['seconds'] for p in pairs); ltime=sum(p['learned']['seconds'] for p in pairs)
    return dict(seed=seed,certified=certified,accepted_counts=[i for i,v in enumerate(relation) if v],
                s1_count=s1_count,s1_mismatches=mismatches,s1_agreement=s1_count>0 and mismatches==0,
                wrong_deletions=wrong,search_requested=search_count,search_available=len(selected),
                search_complete=complete,node_ratio=statistics.median(ratios) if ratios else None,
                ratio_definition='median paired learned/baseline visited nodes, all attempts retained',
                baseline_seconds=btime,learned_seconds=ltime,net_faster=bool(pairs) and ltime<btime,
                pairs=pairs,dev_metrics=evaluate(model,records,steps))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev',nargs='+',required=True); p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True); p.add_argument('--node-limit',type=int,default=100000)
    p.add_argument('--seconds',type=float,default=15); p.add_argument('--steps',type=int,default=16)
    p.add_argument('--search-count',type=int,default=100); a=p.parse_args()
    torch.set_num_threads(1); records=load_records(a.dev,'dev'); results=[]
    for run in a.runs:
        path=Path(run); final=json.loads((path/'final.json').read_text(encoding='utf-8-sig'))
        model=CardinalityBP(final['seed']); t=time.monotonic()
        model.load_state_dict(torch.load(path/'selected.pt',weights_only=True,map_location='cpu')); model.eval()
        load_seconds=time.monotonic()-t
        result=audit(model,records,final['seed'],a.search_count,a.node_limit,a.seconds,a.steps)
        result['load_seconds']=load_seconds
        result['certified']=result['certified'] and final['certifiable_learning_success']
        results.append(result)
    data_counts={str(n):sum(r['n']==n for r in records) for n in (4,9)}
    targets_met=all(v>=200 for v in data_counts.values())
    seeds_valid=sorted(r['seed'] for r in results)==[60101,60102,60103]
    report=dict(continue_experiment=targets_met and seeds_valid and continue_decision(results),dev_counts=data_counts,dev_targets_met=targets_met,seeds_valid=seeds_valid,results=results,code_sha256=code_hashes(),
                environment=dict(python=platform.python_version(),torch=torch.__version__,processor=platform.processor(),threads=1),
                protocol=dict(warmup=0,repetitions=1,search_count=a.search_count,node_limit=a.node_limit,seconds=a.seconds,steps=a.steps))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Sudoku dev gate','',f"Continue: {report['continue_experiment']}",'',
           '| Seed | EO certified | S1 agreement | Wrong bits | S3 attempted | Node ratio | Baseline s | Learned s |',
           '|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['seed']} | {r['certified']} | {r['s1_agreement']} | {r['wrong_deletions']} | {r['search_available']} | {r['node_ratio']} | {r['baseline_seconds']:.4f} | {r['learned_seconds']:.4f} |")
    lines+=['','Missing S3 cases and budget exits prevent a pass. Failed seeds remain in the denominator. No hardware benefit is assumed.']
    (out/'gate.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()

