"""Generate the versioned, offline GitHub artifact command list (no execution)."""
import json
from pathlib import Path


def jobs():
    result=[]
    for n,split,count,seed in ((4,'train',2000,60071),(9,'train',1000,60072),(4,'dev',200,61071),(9,'dev',200,61072)):
        name=f'data_{n}_{split}'
        result.append(dict(name=name,needs=[],command=f'OMP_NUM_THREADS=2 python -m sim.a3_sudoku.gen --n {n} --count {count} --seed {seed} --time-limit 1800 --split {split} --output out/{name}/'))
    data=[r['name'] for r in result]
    for seed in (60101,60102,60103):
        name=f'train_{seed}'
        result.append(dict(name=name,needs=data,command=f'OMP_NUM_THREADS=4 python -m sim.a3_sudoku.train --config sim/a3_sudoku/configs/pilot.json --seed {seed} --train in/data_4_train/manifest.json in/data_9_train/manifest.json --dev in/data_4_dev/manifest.json in/data_9_dev/manifest.json --output out/{name}/',
                           resume=f'OMP_NUM_THREADS=4 python -m sim.a3_sudoku.train --config sim/a3_sudoku/configs/pilot.json --seed {seed} --train in/data_4_train/manifest.json in/data_9_train/manifest.json --dev in/data_4_dev/manifest.json in/data_9_dev/manifest.json --output out/{name}/ --resume out/{name}/last.pt'))
    result.append(dict(name='gate',needs=['data_4_dev','data_9_dev','train_60101','train_60102','train_60103'],
                       command='OMP_NUM_THREADS=1 python -m sim.a3_sudoku.gate --dev in/data_4_dev/manifest.json in/data_9_dev/manifest.json --runs in/train_60101/ in/train_60102/ in/train_60103/ --node-limit 100000 --seconds 8 --search-count 100 --output out/gate/'))
    return dict(schema_version=1,artifact_contract='Upload contents of out/<job>/; download each dependency contents into in/<job>/. For resume restore the entire training output directory, including best checkpoints.',
                environment='Python 3.12, torch 2.14.1 CPU, numpy, scipy preinstalled; no downloads by these commands',
                generation_budget='Four 1800-second jobs = 2 CPU hours. Quota deficits and layer counts must be read from summary.json; never silently fill with duplicates.',jobs=result)

if __name__=='__main__':
    Path(__file__).with_name('jobs_gate.json').write_text(json.dumps(jobs(),indent=2)+'\n',encoding='utf-8')
