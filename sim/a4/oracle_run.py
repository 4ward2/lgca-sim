"""Frozen-only public oracle evaluation, separate from learned comparisons."""
from pathlib import Path
import time
import numpy as np
from sim.logic import to_internal
from sim.cert.tables import extract
from .models import oracle
from .evaluate import run_ac_to_fp
from .data import load_prefix
from .protocol import read_json,validate_freeze,write_json,sha,digest
from .metrics import score_exact
from .certify import certify_prefix

def run(root,out,length,find):
    root=Path(root); out=Path(out)
    frozen=read_json(find(root,'freeze.json')); validate_freeze(frozen,Path.cwd())
    manifest=read_json(find(root,'data_manifest.json'))
    if digest(manifest)!=frozen.get('data_manifest_sha256'): raise ValueError('oracle data manifest differs from freeze')
    hashes={Path(k).name:v for k,v in manifest['source']['files'].items()}
    data=load_prefix(find(root,f'{length}_data.pth').parent,length,role='test',expected_hashes=hashes)
    model=oracle(); predictions=[]; start=time.perf_counter()
    for offset in range(0,len(data['y']),128):
        predictions.append(run_ac_to_fp(model,to_internal(data['x'][offset:offset+128]))['pred'])
    pred=np.concatenate(predictions); sequential=data['x'][:,0].cumsum(1)%2
    if not np.array_equal(pred,sequential) or not np.array_equal(pred,data['y']):
        raise ValueError('oracle mismatch: stop all comparisons')
    write_json(out/'oracle.json',dict(**score_exact(pred,data['y']),length=length,experiment='oracle',seconds=time.perf_counter()-start,sample_id=data['sample_id'].tolist()))
    write_json(out/'certificate.json',certify_prefix(extract(model)))
