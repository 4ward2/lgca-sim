import argparse, json, os, shutil, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sweep

def check_locked(allow):
    if not sweep.common().get('locked', False) and not allow:
        raise SystemExit('common.json must set locked=true (or use --allow-unlocked for dry runs)')

def manifest_tune(task, G, max_minutes):
    c=sweep.common(); jobs=[]
    for arm in ('R','F'):
      for cfg in sweep.tune_grid(arm):
       for seed in c.get('select_seeds',{}).get(task,[0]):
        name=sweep.name(task,arm,G,seed,cfg); args=sweep.train_cmd(task,arm,G,seed,cfg,'out/'+name,4)
        args[0]='python'; args[1]='sim/train.py'
        jobs.append({'name':name,'command':' '.join(args),'upload':[os.path.abspath(HERE)],'download':['out/*'],'max_minutes':max_minutes,'gpu':True,'mirrors':False})
    return jobs
def build(a):
    check_locked(a.allow_unlocked)
    if a.stage=='tune': jobs=manifest_tune(a.task,a.G,a.max_minutes)
    elif a.stage=='final':
      jobs=[]
      for arm in ('R','F','C'):
       if arm=='C' and a.G!=384: continue
       cfg={} if arm=='C' else sweep.selected(a.task,a.G,arm)[0]
       for seed in sweep.FINAL_SEEDS[a.task]:
        n=sweep.name(a.task,arm,a.G,seed,cfg); args=sweep.train_cmd(a.task,arm,a.G,seed,cfg,'out/'+n,4); args[:2]=['python','sim/train.py']
        jobs.append({'name':n,'command':' '.join(args),'upload':[os.path.abspath(HERE)],'download':['out/*'],'max_minutes':a.max_minutes,'gpu':True,'mirrors':False})
    else:
      jobs=[]; fd=os.path.join(ROOT,'runs','final')
      for d in sorted(glob.glob(os.path.join(fd,f'{a.task}_*_G{a.G}_*'))):
       n=os.path.basename(d); args=['python','sim/evaluate.py',n,'--locked','--threads','4']; cmd=' '.join(args)+' && mkdir -p out/'+n+' && cp '+n+'/test.json out/'+n+'/test.json'
       jobs.append({'name':n,'command':cmd,'upload':[d,os.path.abspath(HERE)],'download':['out/*'],'max_minutes':a.max_minutes,'gpu':True,'mirrors':False})
    with open(a.out,'w',encoding='utf-8') as f:
      for j in jobs: f.write(json.dumps(j)+'\n')
    print(f'wrote {len(jobs)} jobs to {a.out}')
def collect(a):
    check_locked(a.allow_unlocked); dest=os.path.join(ROOT,'runs',a.stage); os.makedirs(dest,exist_ok=True); found=[]
    for job in sorted(glob.glob(os.path.join(a.src,'*'))):
      for out in glob.glob(os.path.join(job,'out','*')):
       if not os.path.isdir(out): continue
       n=os.path.basename(out); target=os.path.join(dest,n); os.makedirs(target,exist_ok=True)
       for p in glob.glob(os.path.join(out,'*')):
        q=os.path.join(target,os.path.basename(p));
        if os.path.exists(q) and not a.force and os.path.basename(p) in ('final.json','test.json'): raise SystemExit(f'refusing to overwrite {q}')
        if os.path.isdir(p): shutil.copytree(p,q,dirs_exist_ok=True)
        else: shutil.copy2(p,q)
       if a.stage=='test' and os.path.exists(os.path.join(target,'test.json')):
        tf=os.path.join(ROOT,'runs','final',n); os.makedirs(tf,exist_ok=True); q=os.path.join(tf,'test.json')
        if os.path.exists(q) and not a.force: raise SystemExit(f'refusing to overwrite {q}')
        shutil.copy2(os.path.join(target,'test.json'),q)
       found.append(n)
    print('found',len(found)); print('names:', ' '.join(found))
def main():
 ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='stage',required=True)
 for s in ('tune','final','test'):
  p=sub.add_parser(s); p.add_argument('--task',default='M'); p.add_argument('--G',type=int,default=384); p.add_argument('--out',required=True); p.add_argument('--max-minutes',type=int,default=120); p.add_argument('--allow-unlocked',action='store_true')
  p.set_defaults(func=build)
 p=sub.add_parser('collect'); p.add_argument('--src',required=True); p.add_argument('--stage',choices=('tune','final','test'),required=True); p.add_argument('--force',action='store_true'); p.add_argument('--allow-unlocked',action='store_true'); p.set_defaults(func=collect)
 a=ap.parse_args(); a.func(a)
if __name__=='__main__': main()
