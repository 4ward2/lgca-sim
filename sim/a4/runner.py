"""Frozen evaluation artifacts; no checkpoint selection on OOD."""
from pathlib import Path
import time
import numpy as np
import torch
from sim.logic import to_internal
from sim.cert.tables import extract
from sim.b_arm import TernaryNet
from .protocol import *
from .data import load_prefix
from .models import build,predict,calibrate
from .evaluate import run_ac_to_fp,run_bp,run_dt_streaming,run_logic,run_depth_bp
from .ablations import deploy_variants,flip_unknown,randomize_equal_density
from .metrics import score_exact

@torch.no_grad()
def evaluate_artifact(a,find):
    from .runtime import peak_rss_bytes
    root=Path(a.input); out=Path(a.out)
    frozen=read_json(find(root,'freeze.json')); validate_freeze(frozen,Path.cwd())
    if a.arm not in frozen['gate']['arms'] or a.seed not in SEEDS:
        raise ValueError('evaluation outside frozen model matrix')
    if a.length is None: raise ValueError('length required')
    checkpoint=find(root,'best_soft.pt' if a.arm in ('shared64','unshared64') else 'best_deploy.pt')
    ck=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if ck['seed']!=a.seed or ck['arm']!=a.arm: raise ValueError('checkpoint identity mismatch')
    training_status=read_json(checkpoint.parent/'status.json')
    if training_status.get('status')!='complete':
        raise ValueError('selected checkpoint belongs to incomplete or failed training run')
    model=build(a.arm,a.seed); model.load_state_dict(ck['model']); model.eval()
    raw=find(root,f'{a.length}_data.pth').parent
    manifest=read_json(find(root,'data_manifest.json'))
    if digest(manifest)!=frozen.get('data_manifest_sha256'):
        raise ValueError('evaluation data manifest differs from freeze')
    hashes={Path(k).name:v for k,v in manifest['source']['files'].items()}
    data=load_prefix(raw,a.length,role='test',expected_hashes=hashes)
    train=None
    if a.arm=='b':
        training_path=find(root,'train.npz')
        training_manifest=read_json(find(root,'train_dev_manifest.json'))
        if sha(training_path)!=training_manifest['files']['train.npz']:
            raise ValueError('evaluation calibration artifact hash mismatch')
        with np.load(training_path,allow_pickle=False) as archive: train=dict(archive)
    variants={'main':model}
    if a.arm=='shared64': variants['extended']=model
    if a.arm=='b': variants={'main':calibrate(model,train)}
    if a.command=='ablate':
        if a.arm!='b': raise ValueError('deployment ablations require B')
        if a.kind=='deployment':
            variants=deploy_variants(model,train,checkpoint); variants.pop('source_checkpoint_sha256')
        elif a.kind=='unknown': variants={'control':calibrate(model,train)}; variants['flipped']=flip_unknown(variants['control'])
        elif a.kind=='density':
            parent=calibrate(model,train)
            variants={f'random-{j}':randomize_equal_density(parent,740000+100*(a.seed-SEEDS[0])+j,train) for j in range(10)}
        else: raise ValueError('ablation kind required')
    summaries={}
    for name,m in variants.items():
        predictions=[]; times=[]; state_hashes=[]; curves={}; domain_counts=np.zeros(4,dtype=np.int64); unknown_counts=[0,0]; start=time.perf_counter()
        from .shards import save_shard,load_shard
        identity=dict(checkpoint=sha(checkpoint),freeze=digest(frozen),variant=name,length=a.length)
        shard_root=out/'shards'/name
        for offset in range(0,len(data['y']),128):
            x=data['x'][offset:offset+128]; xv=to_internal(x); T=2*a.length+1
            # AC recomputes state diagnostics on resume; continuous engines can
            # reuse frozen prediction chunks without losing domain metadata.
            cached=load_shard(shard_root,offset,identity,len(x))
            metadata=read_json(shard_root/f'{offset:06d}.json') if cached is not None else {}
            diagnostic=metadata.get('diagnostics')
            if cached is not None and diagnostic is not None:
                predictions.append(cached)
                times.extend(diagnostic['t_fp'])
                state_hashes.extend(diagnostic['state_hashes'])
                domain_counts+=np.asarray(diagnostic['domain_counts'])
                unknown_counts[0]+=diagnostic.get('unknown_counts',[0,0])[0]
                unknown_counts[1]+=diagnostic.get('unknown_counts',[0,0])[1]
                for iteration,counts in diagnostic['curves'].items():
                    target=curves.setdefault(int(iteration),[0,0,0,0])
                    for i,value in enumerate(counts): target[i]+=value
                continue
            batch_times=[]; batch_hashes=[]; batch_domains=np.zeros(4,dtype=np.int64); batch_curves={}; batch_unknown=[0,0]
            # Recompute diagnostics when resuming; shards verify previous
            # predictions and avoid accepting bytes from a different run.
            if isinstance(m,TernaryNet):
                result=run_ac_to_fp(m,xv); times.extend(result['t_fp'].tolist()); state_hashes.append(result['state_hashes'])
                states=(result['domains'][...,0]+2*result['domains'][...,1]).astype(int)
                domain_counts+=np.bincount(states.ravel(),minlength=4)
                batch_times=result['t_fp'].tolist(); batch_hashes=[result['state_hashes']]
                batch_domains=np.bincount(states.ravel(),minlength=4)
                p=result['pred']
                unknown=states==3
                batch_unknown=[int((((p>.5)==data['y'][offset:offset+len(x)]) & unknown).sum()),int(unknown.sum())]
                for i,value in enumerate(batch_unknown): unknown_counts[i]+=value
            elif a.arm=='dt-small':
                result=run_dt_streaming(m,torch.tensor(x)-.5,T); p=result['pred']
            elif a.arm in ('shared64','unshared64'):
                result=run_depth_bp(m,xv,T if name=='extended' else 64); p=result['pred']
            elif a.arm=='b':
                result=run_bp(m,xv,T,hard=name=='qbp'); p=result['pred']
            else:
                result=run_logic(m,xv,T); p=result['pred']
            for iteration,output in result['curve'].items():
                counts=curves.setdefault(iteration,[0,0,0,0])
                correct=(output>.5)==data['y'][offset:offset+len(x)]
                counts[0]+=int(correct.all(1).sum()); counts[1]+=int(correct.sum())
                counts[2]+=len(x); counts[3]+=correct.size
                batch_curves[str(iteration)]=[int(correct.all(1).sum()),int(correct.sum()),len(x),correct.size]
            if cached is not None and not np.array_equal(cached,p): raise ValueError('resumed prediction differs')
            predictions.append(p)
            save_shard(shard_root,offset,p,identity,dict(t_fp=batch_times,state_hashes=batch_hashes,domain_counts=batch_domains.tolist(),curves=batch_curves,unknown_counts=batch_unknown))
        pred=np.concatenate(predictions)
        record=dict(**score_exact(pred,data['y']),arm=a.arm,seed=a.seed,length=a.length,status='complete',
                    experiment=a.kind if a.command=='ablate' else ('extended' if name=='extended' else 'main'),variant=name,
                    sample_id=data['sample_id'].tolist(),config_sha256=digest(frozen),split_sha256=sha(raw/f'{a.length}_data.pth'),
                    checkpoint_sha256=sha(checkpoint),seconds=time.perf_counter()-start,t_fp=times,state_hashes=state_hashes,
                    peak_rss_bytes=peak_rss_bytes(),
                    domain_counts=domain_counts.tolist(),parameters=sum(p.numel() for p in m.parameters()),
                    state_bits_per_cell=2 if isinstance(m,TernaryNet) else (m.regs_per_position() if hasattr(m,'regs_per_position') else None),
                    convergence_failure_rate=0. if isinstance(m,TernaryNet) else None,
                    unknown_count=unknown_counts[1],unknown_conditional_accuracy=unknown_counts[0]/unknown_counts[1] if unknown_counts[1] else None,
                    curve={str(t):dict(exact=c[0]/c[2],bitacc=c[1]/c[3]) for t,c in curves.items()},
                    correct=((pred>.5)==data['y']).all(1).tolist())
        folder=out if a.command=='evaluate' and name=='main' else out/name
        write_json(folder/'result.json',record)
        if isinstance(m,TernaryNet):
            table=extract(m,checkpoint)
            write_json(folder/'table.json',table)
            from .models import calibration_audit
            write_json(folder/'calibration.json',calibration_audit(m,train))
            if a.kind=='density':
                parent_table=extract(parent,checkpoint)
                before=np.concatenate([np.asarray(parent_table[k]).ravel() for k in ('u','r','o')])
                after=np.concatenate([np.asarray(table[k]).ravel() for k in ('u','r','o')])
                write_json(folder/'randomization.json',dict(seed=740000+100*(a.seed-SEEDS[0])+int(name.split('-')[-1]),
                           parent_table_sha256=parent_table['table_sha256'],table_sha256=table['table_sha256'],
                           equal_fraction=float((before==after).mean()),
                           unary_counts=np.asarray(table['u']).sum(-1).tolist(),relation_counts=np.asarray(table['r']).sum((-1,-2)).tolist()))
            from .certify import certify_prefix
            write_json(folder/'certificate.json',certify_prefix(table))
        summaries[name]=record
    if a.kind=='unknown':
        control=summaries['control']; flipped=summaries['flipped']
        if control['state_hashes']!=flipped['state_hashes'] or control['t_fp']!=flipped['t_fp']: raise AssertionError('unknown intervention changed propagation')
    write_json(out/'status.json',dict(status='complete',variants=list(summaries)))
