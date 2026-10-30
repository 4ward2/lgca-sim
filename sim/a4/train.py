"""Remote-only training. Full optimizer and sampling state survive interruption."""
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .models import build,predict,calibrate
from .protocol import digest,write_json
from .evaluate import run_ac_to_fp
from .metrics import score_exact
from sim.logic import to_internal

def require_remote():
    if os.environ.get('GITHUB_ACTIONS')!='true': raise RuntimeError('A4 training is allowed only on GitHub Actions')

def save_checkpoint(path,payload):
    path=Path(path); temp=path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)

def train_run(arm,seed,train,dev,out,resume=None,steps=None,budget_seconds=14400,timing=False):
    require_remote()
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    dt=arm=='dt-small'; total=12000 if dt else 2000
    if steps is not None: total=steps
    model=build(arm,seed); optimizer=torch.optim.Adam(model.parameters(),lr=.001 if dt else .01 if arm in ('r','f') else .03,weight_decay=0)
    generator=torch.Generator().manual_seed(seed); rng=random.Random(seed)
    identity=digest(dict(arm=arm,seed=seed,total=total,ids=train['sample_id'].tolist(),x=train['x'].tolist(),y=train['y'].tolist(),dev_ids=dev['sample_id'].tolist(),dev_x=dev['x'].tolist(),dev_y=dev['y'].tolist()))
    state=dict(step=0,elapsed=0.,best_deploy=-1.,best_soft=-1.,identity=identity)
    if resume:
        ck=torch.load(resume,weights_only=False,map_location='cpu') # own trusted checkpoint, never public dataset
        if ck['identity']!=identity: raise ValueError('resume data/config identity mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        generator.set_state(ck['sampler']); rng.setstate(ck['python_rng']); torch.set_rng_state(ck['torch_rng']); state.update(ck)
    start=time.perf_counter(); initial_elapsed=state['elapsed']; initial_step=state['step']
    size=100 if dt else 128; history=state.get('history',[])
    update_seconds=state.get('update_seconds',[]); dev_seconds=state.get('dev_seconds',[])
    for step in range(initial_step+1,total+1):
        if initial_elapsed+time.perf_counter()-start>=budget_seconds:
            write_json(out/'status.json',dict(status='budget_exceeded',step=step-1,elapsed=initial_elapsed+time.perf_counter()-start))
            return state
        update_start=time.perf_counter()
        if dt:
            epoch=(step-1)//80
            if (step-1)%80==0: state['epoch_order']=torch.randperm(len(train['y']),generator=generator)
            ids=state['epoch_order'][((step-1)%80)*100:((step-1)%80+1)*100]
            lr=.001*min(1,(step/80)/10)*(.1**int(epoch>=60))*(.1**int(epoch>=100))
            for group in optimizer.param_groups: group['lr']=lr
        else: ids=torch.randint(len(train['y']),(size,),generator=generator)
        xv=to_internal(train['x'][ids]); target=torch.tensor(train['y'][ids],dtype=torch.float32)
        T=64 if arm in ('shared64','unshared64') else int(torch.randint(32,65,(1,),generator=generator))
        optimizer.zero_grad(); model.train()
        if dt:
            from .dt_adapter import progressive
            logits,_=progressive(model,torch.tensor(train['x'][ids])-.5,rng)
            loss=torch.nn.functional.cross_entropy(logits,target.long())
        else: loss=torch.nn.functional.binary_cross_entropy(predict(model,arm,xv,T),target)
        if not torch.isfinite(loss):
            state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=generator.get_state(),python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),arm=arm,seed=seed,elapsed=initial_elapsed+time.perf_counter()-start)
            save_checkpoint(out/'failure.pt',state)
            write_json(out/'status.json',dict(status='nonfinite_loss',step=step,arm=arm,seed=seed)); return state
        loss.backward()
        if dt: torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        update_seconds.append(time.perf_counter()-update_start)
        state.update(step=step,elapsed=initial_elapsed+time.perf_counter()-start)
        if step%(1600 if dt else 250)==0 or step==total:
            validation_start=time.perf_counter()
            model.eval(); soft=[]; hard=[]
            ternary=calibrate(model,train) if arm=='b' else None
            with torch.no_grad():
                for offset in range(0,len(dev['y']),128):
                    x=dev['x'][offset:offset+128]; xv=to_internal(x)
                    if dt:
                        from .evaluate import run_dt_streaming
                        p=run_dt_streaming(model,torch.tensor(x)-.5,65)['pred']; q=p
                    else:
                        p=predict(model,arm,xv,64 if arm in ('shared64','unshared64') else 65).numpy()
                        q=run_ac_to_fp(ternary,xv)['pred'] if ternary else (predict(model,arm,xv,65,True).numpy() if arm in ('r','f') else p)
                    soft.append(p); hard.append(q)
            soft_score=score_exact(np.concatenate(soft),dev['y'])['exact']; deploy_score=score_exact(np.concatenate(hard),dev['y'])['exact']
            history.append(dict(step=step,soft=soft_score,deploy=deploy_score,loss=float(loss)))
            for name,value in [('best_soft',soft_score),('best_deploy',deploy_score)]:
                if value>state[name]:
                    state[name]=value; save_checkpoint(out/(name+'.pt'),dict(model=model.state_dict(),step=step,arm=arm,seed=seed))
                    if name=='best_deploy' and ternary:
                        from sim.cert.tables import extract
                        write_json(out/'table.json',extract(ternary,out/(name+'.pt')))
            write_json(out/'history.json',history)
            dev_seconds.append(time.perf_counter()-validation_start)
        state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=generator.get_state(),python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),arm=arm,seed=seed,history=history,update_seconds=update_seconds,dev_seconds=dev_seconds,elapsed=initial_elapsed+time.perf_counter()-start)
        # Every update is resumable; elapsed time includes selection and checkpoint overhead on next update.
        save_checkpoint(out/'last.pt',state)
        if initial_elapsed+time.perf_counter()-start>=budget_seconds:
            write_json(out/'status.json',dict(status='budget_exceeded',step=step,elapsed=state['elapsed'])); return state
    write_json(out/'status.json',dict(status='complete',step=total,elapsed=state['elapsed'],arm=arm,seed=seed))
    if timing:
        from .runtime import peak_rss_bytes
        if len(update_seconds)!=60: raise ValueError('timing pilot requires ten warmups and fifty measured updates')
        write_json(out/'timing.json',dict(arm=arm,seed=seed,warmup_updates=10,measured_updates=50,
                   seconds_per_update=float(np.mean(update_seconds[10:])),dev_seconds=float(np.mean(dev_seconds)),
                   evaluation_runner_hours=None,evaluation_status='requires synthetic long-length timing',
                   peak_rss_bytes=peak_rss_bytes(),
                   elapsed=state['elapsed']))
    return state
