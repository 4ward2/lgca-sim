"""Pure prerequisite checks; generating a candidate never locks an experiment."""
import math
from .protocol import ARMS,PILOTS,SEEDS,LENGTHS,digest

def pilot_gate(timings,learnability,environment,checker):
    issues=[]; indexed={}
    for record in timings:
        arm=record.get('arm')
        if arm in indexed or arm not in ARMS:
            issues.append('duplicate/unknown timing arm'); continue
        indexed[arm]=record
        if record.get('budget_qualified') is not True:
            issues.append(f'{arm}: evaluation budget not qualified')
        for field in ('seconds_per_update','dev_seconds','evaluation_runner_hours'):
            value=record.get(field)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                issues.append(f'{arm}: invalid {field}')
        if record.get('warmup_updates')!=10 or record.get('measured_updates')!=50:
            issues.append(f'{arm}: incomplete timing protocol')
    if set(indexed)!=set(ARMS): issues.append('missing timing arm')
    rows={}
    for record in learnability:
        seed=record.get('seed')
        if seed not in PILOTS or seed in rows: issues.append('duplicate/unregistered B pilot')
        rows[seed]=record
    if set(rows)!=set(PILOTS): issues.append('missing B learnability pilot')
    passed=sum(r.get('status')=='complete' and isinstance(r.get('dev_ac_exact'),(int,float)) and math.isfinite(r['dev_ac_exact']) and r['dev_ac_exact']>=.99 for r in rows.values())
    if passed<2: issues.append('B learnability gate failed')
    if not environment.get('wheel_hashes') or not isinstance(environment['wheel_hashes'],dict): issues.append('missing wheel hashes')
    if checker.get('status')!='qualified' or not checker.get('source_sha256') or not checker.get('mutation_tests_passed'): issues.append('checker not qualified')
    estimates={}
    if not any('invalid' in x for x in issues) and set(indexed)==set(ARMS):
        for arm,r in indexed.items():
            updates=12000 if arm=='dt-small' else 2000
            validations=8
            estimates[arm]=(updates*r['seconds_per_update']+validations*r['dev_seconds'])/3600
        enabled=[a for a in ARMS if a!='dt-small' or estimates[a]<=4]
        if any(estimates[a]>4 for a in enabled): issues.append('required training run exceeds four hours')
        # Explicit reserves cover fetch/data/smoke, certificates, reporting,
        # three B learnability runs and the six timing pilots. Do not hide
        # these behind concurrency or count only successful formal training.
        overhead_hours=2+2+1+9+3
        total=overhead_hours+sum(10*(estimates[a]+indexed[a]['evaluation_runner_hours']) for a in enabled)
        if total>120: issues.append('complete matrix exceeds 120 runner-hours')
    else: enabled=[]; total=None
    return dict(status='READY_FOR_REVIEW' if not issues else 'STOP',issues=issues,arms=enabled,
                training_hours_per_seed=estimates,total_runner_hours=total,passed_b_pilots=passed,
                dt_status='enabled' if 'dt-small' in enabled else 'not-run-budget')

def freeze_candidate(timings,learnability,environment,checker,files):
    gate=pilot_gate(timings,learnability,environment,checker)
    return dict(status='DRAFT',seeds=SEEDS,lengths=LENGTHS,files=files,gate=gate,
                evidence_sha256=digest(dict(timings=timings,learnability=learnability,environment=environment,checker=checker)),
                launch_allowed=False)
