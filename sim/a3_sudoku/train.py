"""Fixed answer-supervised CPU recipe. Only tiny unit/benchmark runs are local."""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
import torch
from .soft import CardinalityBP, batched_forward
from .graph import build_standard
from .rounding import cardinality_to_relation, certify_eo
from .domains import init_domains,synchronous_closure,read_solution
from .manifest import replay_manifest,code_hashes,digest
from .data import _seed
from .oracle import validate_solution


def load_records(paths,split):
    records=[]; seen=set()
    for path in paths:
        for r in replay_manifest(json.loads(Path(path).read_text(encoding='utf-8-sig'))):
            key=(r['n'],tuple(r['puzzle']))
            if r['split']==split and key not in seen:
                seen.add(key); records.append(r)
    return records


def supervised_loss(history,solutions,puzzles,n):
    target=torch.as_tensor(solutions,dtype=torch.long)
    blank=torch.as_tensor(puzzles)<0
    if not blank.any():
        return history[-1][torch.isfinite(history[-1])].sum()*0
    total=0
    for t,belief in enumerate(history,1):
        logits=(belief[...,1]-belief[...,0]).reshape(-1,n*n,n)[blank]
        labels=target[blank]
        ce=torch.nn.functional.cross_entropy(logits,labels)
        truth=torch.nn.functional.one_hot(labels,n).bool()
        bce=.5*(torch.nn.functional.softplus(-logits[truth]).mean()+torch.nn.functional.softplus(logits[~truth]).mean())
        total=total+(ce+.2*bce)*(2*t/(len(history)*(len(history)+1)))
    return total


@torch.no_grad()
def evaluate(model,records,steps=16):
    relation=cardinality_to_relation(model.theta)
    metrics=dict(count=len(records),soft_legal=0,soft_exact=0,rounded_legal=0,rounded_exact=0,unknown=0,contradiction=0,
                 accepted_counts=[i for i,v in enumerate(relation) if v],certified=certify_eo(relation))
    for r in records:
        g=build_standard(r['n']); p=r['puzzle']
        belief=batched_forward(model,g,[p],steps)[0]
        logits=(belief[:,1]-belief[:,0]).reshape(-1,g.n)
        answer=logits.argmax(-1).tolist()
        metrics['soft_legal']+=int(validate_solution(answer,g.n,p))
        metrics['soft_exact']+=int(answer==list(r['solution']))
        d=synchronous_closure(g,init_domains(g,p),relation=relation)
        answer=read_solution(g,d,p)
        metrics['rounded_legal']+=int(answer is not None)
        metrics['rounded_exact']+=int(answer==list(r['solution']))
        metrics['contradiction']+=int(0 in d)
        metrics['unknown']+=int(answer is None and 0 not in d)
    return metrics


def select_checkpoint(previous,metrics,step):
    # Never silently promote an uncertified table to the certified best.
    if not metrics['certified']:
        return False
    return previous is None or metrics['rounded_legal']>previous['metrics']['rounded_legal']


def train(config,train_records,dev_records,output,resume=None):
    _seed(config['seed']); torch.set_num_threads(config.get('threads',4))
    torch.use_deterministic_algorithms(True); torch.manual_seed(config['seed'])
    rng=random.Random(config['seed']); model=CardinalityBP(config['seed'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    run_config=dict(config,code_sha256=code_hashes(),train_sha256=digest(train_records),dev_sha256=digest(dev_records))
    run_config['config_sha256']=digest(run_config)
    best=None; best_soft=None; step=0; elapsed=0.
    if resume:
        saved=torch.load(resume,weights_only=False,map_location='cpu')
        if saved['config']!=run_config:
            raise ValueError('resume config/data/code mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng']); torch.set_rng_state(saved['torch_rng'])
        best=saved['best']; best_soft=saved['best_soft']; step=saved['step']; elapsed=saved['elapsed']
    (out/'config.json').write_text(json.dumps(run_config,indent=2),encoding='utf-8')
    pools=defaultdict(list)
    for r in train_records:
        pools[r['n']].append(r)
    if not pools[4] or not dev_records:
        raise ValueError('4x4 train and nonempty dev required')
    start=time.monotonic(); status='complete'; metrics=None
    def state():
        return dict(config=run_config,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.getstate(),
                    torch_rng=torch.get_rng_state(),best=best,best_soft=best_soft,step=step,elapsed=elapsed+time.monotonic()-start)
    while step<config['steps']:
        if elapsed+time.monotonic()-start>=config['time_limit']:
            status='budget'; break
        n=4 if step<config.get('warmup_steps',500) or not pools[9] else rng.choice([4,9])
        iterations=rng.choice([8,16] if step<config.get('warmup_steps',500) else [16,32,64])
        layers=defaultdict(list)
        for r in pools[n]:
            layers[r.get('level','UNKNOWN')].append(r)
        keys=sorted(layers)
        batch=[rng.choice(layers[rng.choice(keys)]) for _ in range(config['batch'])]
        optimizer.zero_grad()
        _,history=batched_forward(model,build_standard(n),[r['puzzle'] for r in batch],iterations,True)
        loss=supervised_loss(history,[r['solution'] for r in batch],[r['puzzle'] for r in batch],n)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True); optimizer.step()
        step+=1
        if step%config['eval_every']==0 or step==config['steps']:
            metrics=evaluate(model,dev_records,config.get('eval_steps',16))
            if select_checkpoint(best,metrics,step):
                best=dict(step=step,metrics=metrics); torch.save(model.state_dict(),out/'best.pt')
            if best_soft is None or metrics['soft_legal']>best_soft['metrics']['soft_legal']:
                best_soft=dict(step=step,metrics=metrics); torch.save(model.state_dict(),out/'best_soft.pt')
            with (out/'log.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(step=step,loss=float(loss.detach()),metrics=metrics,seconds=elapsed+time.monotonic()-start))+'\n')
        torch.save(state(),out/'last.tmp'); (out/'last.tmp').replace(out/'last.pt')
    # Final diagnostics include budget-stopped seeds; no oracle substitution.
    metrics=evaluate(model,dev_records,config.get('eval_steps',16))
    selected=best
    if best is not None:
        model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    selected_metrics=evaluate(model,dev_records,config.get('eval_steps',16))
    final=dict(status=status,step=step,seconds=elapsed+time.monotonic()-start,last_metrics=metrics,
               selected=selected,metrics=selected_metrics,certifiable_learning_success=best is not None,
               theta=model.theta.detach().tolist(),beta=model.beta.detach().tolist(),seed=config['seed'],
               available_layers={str(n):sorted({r.get('level','UNKNOWN') for r in rs}) for n,rs in pools.items()})
    torch.save(model.state_dict(),out/'selected.pt')
    (out/'final.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    (out/'table.json').write_text(json.dumps(dict(theta=final['theta'],relation=cardinality_to_relation(final['theta']),sha256=digest(final['theta']))),encoding='utf-8')
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--train',nargs='+',required=True)
    p.add_argument('--dev',nargs='+',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int); p.add_argument('--resume')
    a=p.parse_args(); config=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    if a.seed is not None:
        config['seed']=a.seed
    training=load_records(a.train,'train'); dev=load_records(a.dev,'dev')
    if any(not any(r['n']==n for r in rs) for rs in (training,dev) for n in (4,9)):
        raise ValueError('pilot requires nonempty 4x4 and 9x9 train/dev; inspect generation deficits')
    families={(r['n'],r['family']) for r in training}
    if any((r['n'],r['family']) in families for r in dev):
        raise ValueError('train/dev solution-family leakage')
    print(json.dumps(train(config,training,dev,a.output,a.resume)))

if __name__=='__main__':
    main()
