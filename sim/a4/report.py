"""Fail-closed artifact admission. Never selects models using test scores."""
from pathlib import Path
import csv
import math
from .protocol import read_json,write_json

def validate_records(plan,records):
    required={tuple(x) for x in plan}; admitted={}; issues=[]
    for r in records:
        if not isinstance(r,dict): issues.append('record is not an object'); continue
        try:
            key=(r['arm'],r['seed'],r['length'])
            if key not in required: raise ValueError('unplanned record')
            if key in admitted: raise ValueError('duplicate record')
            if r['status']!='complete': raise ValueError('incomplete/failed run')
            if len(r['sample_id'])!=10000 or sorted(r['sample_id'])!=list(range(10000)): raise ValueError('truncated or duplicate sample IDs')
            if not all(isinstance(r[x],(int,float)) and math.isfinite(r[x]) and 0<=r[x]<=1 for x in ('exact','bitacc')): raise ValueError('invalid metric')
            if not r.get('config_sha256') or not r.get('split_sha256'): raise ValueError('missing identity')
            admitted[key]=r
        except (KeyError,TypeError,ValueError) as e: issues.append(str(e))
    issues += ['missing '+str(k) for k in sorted(required-set(admitted))]
    for length in {k[2] for k in admitted}:
        rows=[r for k,r in admitted.items() if k[2]==length]
        if len({r['split_sha256'] for r in rows})>1: issues.append('different split hashes')
        if len({r['config_sha256'] for r in rows})>1: issues.append('different config hashes')
    return dict(headline='protocol failure' if issues else 'complete',issues=issues,admitted=len(admitted),required=len(required))

def report(root,out,plan,expected_config_sha256=None):
    records=[]; issues=[]
    from .attempts import resolve_attempts
    for path in Path(root).rglob('attempts.json'):
        try: resolve_attempts(read_json(path))
        except (ValueError,TypeError,KeyError) as e: issues.append('attempt chain: '+str(e))
    for path in Path(root).rglob('result.json'):
        try: records.append(read_json(path))
        except (ValueError,OSError) as e: issues.append(str(e))
    if expected_config_sha256 is not None:
        for row in records:
            if isinstance(row,dict) and row.get('config_sha256')!=expected_config_sha256:
                issues.append('result config differs from reviewed freeze')
    main_records=[r for r in records if not isinstance(r,dict) or r.get('experiment','main')=='main']
    result=validate_records(plan,main_records); result['issues']+=issues
    ablations=[r for r in records if isinstance(r,dict) and r.get('experiment','main') not in ('main','extended','oracle')]
    result['ablation_records']=len(ablations)
    expected_variants={'deployment':{'soft','qbp','ac'},'unknown':{'control','flipped'},'density':{f'random-{j}' for j in range(10)}}
    identities=set()
    for r in ablations:
        key=(r.get('experiment'),r.get('variant'),r.get('seed'),r.get('length'))
        if key in identities: result['issues'].append('duplicate ablation '+str(key))
        identities.add(key)
        if r.get('variant') not in expected_variants.get(r.get('experiment'),set()): result['issues'].append('unregistered ablation '+str(key))
        if r.get('status')!='complete' or r.get('sample_id')!=list(range(10000)): result['issues'].append('incomplete ablation '+str(key))
        matching=[m for m in main_records if isinstance(m,dict) and m.get('arm')=='b' and m.get('seed')==r.get('seed') and m.get('length')==r.get('length')]
        if len(matching)!=1 or any(r.get(field)!=matching[0].get(field) for field in ('sample_id','config_sha256','split_sha256','checkpoint_sha256')):
            result['issues'].append('ablation pairing identity mismatch '+str(key))
        if any(not isinstance(r.get(field),(int,float)) or not math.isfinite(r[field]) or not 0<=r[field]<=1 for field in ('exact','bitacc')):
            result['issues'].append('invalid ablation metric '+str(key))
    b_keys=[(s,l) for arm,s,l in plan if arm=='b']
    for seed,length in b_keys:
        for kind,variants in expected_variants.items():
            for variant in variants:
                if (kind,variant,seed,length) not in identities: result['issues'].append(f'missing ablation {kind}/{variant}/{seed}/{length}')
    extended=[r for r in records if isinstance(r,dict) and r.get('experiment')=='extended']
    expected_extended={(s,l) for arm,s,l in plan if arm=='shared64'}
    actual_extended=[(r.get('seed'),r.get('length')) for r in extended]
    if set(actual_extended)!=expected_extended or len(actual_extended)!=len(expected_extended):
        result['issues'].append('missing/duplicate shared64 extended evaluation')
    for row in extended:
        peers=[r for r in main_records if isinstance(r,dict) and r.get('arm')=='shared64' and r.get('seed')==row.get('seed') and r.get('length')==row.get('length')]
        if len(peers)!=1 or any(row.get(k)!=peers[0].get(k) for k in ('sample_id','checkpoint_sha256','config_sha256','split_sha256')):
            result['issues'].append('shared64 extended identity mismatch')
        if row.get('status')!='complete' or any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in ('exact','bitacc')):
            result['issues'].append('invalid shared64 extended result')
    oracle_paths=list(Path(root).rglob('oracle.json'))
    oracle_lengths=[]
    for path in oracle_paths:
        try:
            oracle=read_json(path)
            if oracle.get('exact')!=1 or oracle.get('count')!=10000 or oracle.get('sample_id')!=list(range(10000)):
                raise ValueError('oracle integrity mismatch')
            oracle_lengths.append(oracle['length'])
        except (ValueError,KeyError,AttributeError) as e: result['issues'].append(str(e))
    expected_lengths={l for _,_,l in plan}
    if set(oracle_lengths)!=expected_lengths or len(oracle_lengths)!=len(expected_lengths):
        result['issues'].append('missing/duplicate oracle length')
    for path in Path(root).rglob('table.json'):
        certificate=path.with_name('certificate.json')
        try:
            from .certify import replay
            if not certificate.exists(): raise ValueError('missing certificate: '+str(path))
            replay(read_json(path),read_json(certificate))
        except (ValueError,KeyError,OSError) as e: result['issues'].append(str(e))
    for path in Path(root).rglob('result.json'):
        try:
            row=read_json(path)
            if not isinstance(row,dict): continue
            discrete=row.get('arm')=='b' and (row.get('experiment','main')=='main' or
                     row.get('experiment') in ('density','unknown') or row.get('variant')=='ac')
            if discrete and not path.with_name('table.json').is_file():
                result['issues'].append('missing discrete table: '+str(path))
        except (ValueError,OSError): pass
    if result['issues']: result['headline']='protocol failure'
    if issues: result['headline']='protocol failure'
    from .metrics import aggregate_seeds,paired_bootstrap
    groups={}
    for row in main_records:
        if isinstance(row,dict) and row.get('status')=='complete':
            groups.setdefault((row.get('arm'),row.get('length')),[]).append(row)
    result['descriptive']={}
    for (arm,length),rows in groups.items():
        expected=[s for a,s,l in plan if a==arm and l==length]
        try: result['descriptive'][f'{arm}/{length}']=aggregate_seeds(rows,expected)
        except ValueError: pass
    result['deployment_intervals']={}
    result['control_intervals']={}
    from .protocol import SEEDS
    for length in {l for _,_,l in plan}:
        for variant in ('soft','qbp'):
            selected={(r['seed'],r['variant']):r['exact'] for r in ablations
                      if r.get('experiment')=='deployment' and r.get('length')==length}
            if all((s,v) in selected for s in SEEDS for v in ('ac',variant)):
                result['deployment_intervals'][f'{length}/{variant}-minus-ac']=paired_bootstrap(
                    [selected[s,'ac'] for s in SEEDS],[selected[s,variant] for s in SEEDS])
        density={(r['seed'],r['variant']):r['exact'] for r in ablations
                 if r.get('experiment')=='density' and r.get('length')==length}
        main={r['seed']:r['exact'] for r in main_records if isinstance(r,dict) and r.get('arm')=='b' and r.get('length')==length}
        if all(s in main and all((s,f'random-{j}') in density for j in range(10)) for s in SEEDS):
            random_means=[sum(density[s,f'random-{j}'] for j in range(10))/10 for s in SEEDS]
            result['control_intervals'][f'{length}/density-minus-trained']=paired_bootstrap([main[s] for s in SEEDS],random_means)
        unknown={(r['seed'],r['variant']):r['exact'] for r in ablations
                 if r.get('experiment')=='unknown' and r.get('length')==length}
        if all((s,v) in unknown for s in SEEDS for v in ('control','flipped')):
            result['control_intervals'][f'{length}/unknown-flipped-minus-control']=paired_bootstrap(
                [unknown[s,'control'] for s in SEEDS],[unknown[s,'flipped'] for s in SEEDS])
    out=Path(out); write_json(out/'summary.json',result); write_json(out/'missing.json',result['issues'])
    write_json(out/'provenance.json',dict(inputs=[str(p.relative_to(root)) for p in sorted(Path(root).rglob('result.json'))],selection='none; report consumes frozen outputs'))
    with (out/'per_seed.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm','seed','length','status','exact','bitacc'])
        writer.writeheader()
        for row in records:
            if isinstance(row,dict): writer.writerow({k:row.get(k,'') for k in writer.fieldnames})
    lines=['# A4','',result['headline'],'','| Arm/length | Mean | SD | Median | Min | Max |','|---|---:|---:|---:|---:|---:|']
    for key,row in sorted(result['descriptive'].items()):
        lines.append('| '+key+' | '+' | '.join(str(row[field]) for field in ('mean','sd','median','min','max'))+' |')
    lines+=['','Primary endpoint: L512 exact_final. Missing records prevent a confirmatory verdict.','']
    lines+=result['issues']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
