"""Self-contained, hashed replay manifests and explicitly budgeted offline CLI."""
import argparse
import hashlib
import json
import time
import multiprocessing
import tempfile
from pathlib import Path
from .data import parse_pairs, family_split, augment, _seed
from .oracle import validate_solution


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def code_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def make_manifest(splits,seed=60000,copies=1,provenance=None):
    _seed(seed)
    if copies<1:
        raise ValueError('positive copies required')
    records=[]; order=[]
    for split in ('train','dev','test'):
        for record in splits[split]:
            for copy in range(copies):
                r=augment(record,seed+len(order))
                r['sample_id']=digest([split,record['source'],record['row_id'],copy,r['puzzle'],r['solution']])
                records.append(r); order.append(r['sample_id'])
    records=json.loads(json.dumps(records))
    manifest=dict(schema_version=1,seed=seed,copies=copies,records=records,order=order,
                  provenance=provenance or {},code_sha256=code_hashes(),
                  symmetry_scope='full 4x4; digit+dihedral only for larger N; not complete equivalence exclusion')
    manifest['sha256']=digest(manifest)
    return manifest


def replay_manifest(manifest):
    payload={k:v for k,v in manifest.items() if k!='sha256'}
    if digest(payload)!=manifest.get('sha256') or manifest.get('schema_version')!=1:
        raise ValueError('manifest checksum/schema mismatch')
    records=manifest['records']
    if [r['sample_id'] for r in records]!=manifest['order']:
        raise ValueError('manifest order mismatch')
    for r in records:
        if not validate_solution(r['solution'],r['n'],r['puzzle']):
            raise ValueError('invalid manifest pair')
    return records


def _build(args,temporary):
    records=parse_pairs(args.input,args.n,source=args.source_url,uniqueness_seconds=args.uniqueness_seconds,max_records=args.max_records)
    splits=family_split(records,args.seed)
    manifest=make_manifest(splits,args.seed,args.copies,dict(url=args.source_url,revision=args.revision,license=args.license,
                                                         input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()))
    Path(temporary).write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--n',type=int,default=9)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--max-records',type=int,required=True)
    ap.add_argument('--seconds',type=float,required=True)
    ap.add_argument('--uniqueness-seconds',type=float,required=True)
    ap.add_argument('--copies',type=int,default=1)
    ap.add_argument('--source-url',required=True)
    ap.add_argument('--revision',required=True)
    ap.add_argument('--license',required=True)
    args=ap.parse_args()
    if args.seconds<=0:
        ap.error('seconds must be positive')
    _seed(args.seed)
    # A child gives a real wall-clock cap, including expensive canonicalization.
    output=Path(args.output)
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.a3-tmp',delete=False) as f:
        temporary=f.name
    worker=multiprocessing.get_context('spawn').Process(target=_build,args=(args,temporary))
    try:
        worker.start()
        worker.join(args.seconds)
        if worker.is_alive():
            worker.terminate(); worker.join()
            raise TimeoutError('overall budget exhausted; no output published')
        if worker.exitcode!=0:
            raise RuntimeError('manifest worker failed; no output published')
        Path(temporary).replace(output)
        print(json.dumps(dict(output=str(output),status='complete')))
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__=='__main__':
    main()
