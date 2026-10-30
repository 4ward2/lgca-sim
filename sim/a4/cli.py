"""Uniform artifact-only command boundary; unavailable evidence fails closed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
import torch
from .protocol import *

def find(root,name):
    paths=list(Path(root).rglob(name))
    if len(paths)!=1: raise ValueError(f'expected one {name}, found {len(paths)}')
    return paths[0]

class Tee:
    def __init__(self, stream, log): self.stream, self.log = stream, log
    def write(self, text):
        self.stream.write(text); self.log.write(text); self.log.flush()
        return len(text)
    def flush(self): self.stream.flush(); self.log.flush()

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['fetch','data','smoke','pilot','freeze','train','evaluate','ablate','certify','maze-gate','report'])
    parser.add_argument('--input',required=True); parser.add_argument('--out',required=True); parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--suite',default='prefix'); parser.add_argument('--arm',choices=ARMS+['oracle'],default='b'); parser.add_argument('--seed',type=int,default=PILOTS[0])
    parser.add_argument('--length',type=int,choices=LENGTHS); parser.add_argument('--kind',choices=['deployment','density','unknown']); parser.add_argument('--resume')
    parser.add_argument('--learnability',action='store_true')
    args=parser.parse_args(argv); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(args.threads)
    try:
        with (out/'run.log').open('a',encoding='utf-8') as log:
            with redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
                print('A4 command:',args.command)
                try: execute(args)
                except Exception as error:
                    print(str(error),file=sys.stderr)
                    raise
    except Exception as e:
        write_json(out/'status.json',dict(status='failed',reason=str(e),command=args.command)); print(str(e),file=sys.stderr); return 1
    status_path=out/'status.json'
    if status_path.exists() and read_json(status_path).get('status') in ('nonfinite_loss','budget_exceeded','algorithm_failure'):
        return 2
    return 0

def execute(a):
    out=Path(a.out); root=Path(a.input)
    if a.command=='fetch':
        from .sources import BASE,fetch_archive
        from .train import require_remote
        require_remote()
        filename='prefix_sums_data.tar.gz' if a.suite=='prefix' else 'maze_data_train_9.tar.gz'
        expected=read_json(find(root,'source_pin.json'))['sha256'] if list(root.rglob('source_pin.json')) else None
        try:
            fetch_archive(BASE+filename,out,expected)
        except OSError as error:
            if a.suite!='prefix': raise
            from .data import regenerate_prefix
            fallback=out/'raw'/'prefix_sums_data'; fallback.mkdir(parents=True,exist_ok=True)
            for length in [32]+LENGTHS:
                seed=731032 if length==32 else 732000+length
                x,y=regenerate_prefix(length,seed)
                torch.save(x,fallback/f'{length}_data.pth'); torch.save(y,fallback/f'{length}_targets.pth')
            write_json(out/'manifest.json',dict(source='ETH-prefix-regenerated-v1',reason=str(error),generator_commit=DATA_SHA,
                       distribution_equivalent=True,sample_identical=False,
                       files={p.as_posix():sha(p) for p in fallback.glob('*.pth')}))
        return
    if a.command=='data':
        from .data import load_prefix,calibration
        raw=find(root,'32_data.pth').parent
        train=load_prefix(raw,32,role='train'); dev=load_prefix(raw,32,role='dev')
        import numpy as np
        for role,data in [('train',train),('dev',dev)]:
            folder=out/'train-dev'; folder.mkdir(exist_ok=True)
            np.savez(folder/(role+'.npz'),**data)
        write_json(out/'train-dev'/'train_dev_manifest.json',dict(files={p.name:sha(p) for p in (out/'train-dev').glob('*.npz')},calibration_ids=calibration(train)['sample_id'].tolist()))
        train_words={row.tobytes() for row in train['x'][:,0]}
        dev_words={row.tobytes() for row in dev['x'][:,0]}
        write_json(out/'data_audit.json',dict(train_count=len(train['y']),dev_count=len(dev['y']),
                   train_unique=len(train_words),dev_unique=len(dev_words),overlap_unique=len(train_words & dev_words),
                   overlap_dev_examples=sum(row.tobytes() in train_words for row in dev['x'][:,0]),
                   split_policy='retain official split; disclose overlap'))
        # Test bytes stay in the upstream artifact. No test tensor is opened here.
        write_json(out/'data_manifest.json',dict(source=read_json(find(root,'manifest.json')),train_dev_sha256=sha(out/'train-dev'/'train_dev_manifest.json'))); return
    if a.command=='smoke':
        if a.suite=='checker':
            from .qualification import qualify
            qualify(out); return
        if a.suite=='resume':
            from .resume_check import verify
            verify(out); return
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s','sim/a4/tests','-v'],capture_output=True,text=True)
        (out/'tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError('smoke tests failed')
        from .environment import environment_lock,fetch_cpu_wheels
        wheel_root=os.environ.get('A4_WHEEL_DIRECTORY')
        if not wheel_root and os.environ.get('GITHUB_ACTIONS')=='true':
            wheel_root=out/'wheels'; fetch_cpu_wheels(wheel_root)
        write_json(out/'environment.json',environment_lock(wheel_root))
        return
    if a.command in ('train','pilot'):
        import numpy as np
        from .train import train_run
        from .artifacts import load_training_artifact
        train,dev=load_training_artifact(root)
        if a.command=='train':
            frozen=read_json(find(root,'freeze.json'))
            validate_freeze(frozen,Path.cwd())
            if a.arm not in frozen['gate']['arms']:
                raise ValueError('arm excluded by frozen budget gate')
            if a.seed not in SEEDS: raise ValueError('formal training requires formal seed')
        start=time.perf_counter()
        if a.learnability and (a.command!='pilot' or a.arm!='b' or a.seed not in PILOTS):
            raise ValueError('learnability requires a registered B pilot')
        if a.learnability:
            pilot_timing=read_json(find(root,'timing.json'))
            estimate=(2000*pilot_timing['seconds_per_update']+8*pilot_timing['dev_seconds'])/3600
            if estimate>3: raise ValueError('B learnability pilot exceeds three-hour gate')
        state=train_run(a.arm,a.seed,train,dev,out,a.resume,steps=2000 if a.learnability else 60 if a.command=='pilot' else None,timing=a.command=='pilot' and not a.learnability,budget_seconds=10800 if a.learnability else 14400)
        if a.learnability:
            status=read_json(out/'status.json')
            write_json(out/'learnability.json',dict(seed=a.seed,status=status['status'],dev_ac_exact=state['best_deploy'] if state else None))
        if a.command=='pilot' and not a.learnability and (out/'timing.json').exists():
            from .models import build
            from .benchmark import synthetic_latency
            model=build(a.arm,a.seed); model.load_state_dict(state['model'])
            latency=synthetic_latency(model,a.arm,train)
            write_json(out/'latency.json',latency)
            timing=read_json(out/'timing.json'); timing['evaluation_runner_hours']=latency['evaluation_runner_hours']
            timing['evaluation_status']='synthetic measured estimate'
            timing['budget_qualified']=latency['budget_qualified']
            write_json(out/'timing.json',timing)
        return
    if a.command=='freeze':
        from .gates import freeze_candidate
        timings=[read_json(p) for p in root.rglob('timing.json')]
        learnability=[read_json(p) for p in root.rglob('learnability.json')]
        environment=read_json(find(root,'environment.json'))
        qualification=read_json(find(root,'checker_qualification.json'))
        if read_json(find(root,'resume_check.json')).get('status')!='pass':
            raise ValueError('optimizer trajectory resume qualification failed')
        from sim.cert.tables import provenance
        if qualification.get('source_sha256')!=provenance()['checker_sha256']:
            raise ValueError('checker qualification does not bind current checker sources')
        source_paths=list(Path('sim/a4').rglob('*'))+list(Path('sim/cert').glob('*.py'))+[Path('sim/b_arm.py'),Path('sim/logic.py')]
        files={p.as_posix():sha(p) for p in sorted(source_paths) if p.is_file() and '__pycache__' not in p.parts}
        candidate=freeze_candidate(timings,learnability,environment,qualification,files)
        data_manifest=read_json(find(root,'data_manifest.json'))
        candidate['data_manifest_sha256']=digest(data_manifest)
        candidate['environment_sha256']=digest(environment)
        candidate['checker_sha256']=qualification['source_sha256']
        write_json(out/'freeze.json',candidate); write_json(out/'go.json',candidate['gate'])
        if candidate['gate']['status']!='READY_FOR_REVIEW': raise ValueError('pilot prerequisites failed: '+str(candidate['gate']['issues']))
        from .planning import materialize
        write_json(out/'jobs_a4.json',materialize(read_json('sim/a4/jobs_a4.json'),candidate))
        return
    if a.command=='certify':
        from .certify import certify_prefix
        write_json(out/'certificate.json',certify_prefix(read_json(find(root,'table.json')))); return
    if a.command=='maze-gate':
        import numpy as np
        from .maze import maze_gate
        from .wave_cert import wave_certificate,replay
        proof=wave_certificate(); replay(proof)
        result=maze_gate(np.load(find(root,'inputs.npy')),np.load(find(root,'solutions.npy')))
        write_json(out/'gate.json',result); write_json(out/'wave.json',proof); return
    if a.command=='report':
        from .report import report
        frozen=read_json(find(root,'freeze.json'))
        validate_freeze(frozen,Path.cwd())
        plan=[(arm,s,l) for arm in frozen['gate']['arms'] for s in SEEDS for l in LENGTHS]
        result=report(root,out,plan,expected_config_sha256=digest(frozen))
        if result['issues']: raise ValueError('incomplete evidence package')
        return
    if a.command in ('evaluate','ablate'):
        if a.arm=='oracle':
            from .oracle_run import run
            run(root,out,a.length,find); return
        from .runner import evaluate_artifact
        evaluate_artifact(a,find)
        return
    raise RuntimeError('unhandled command: '+a.command)
