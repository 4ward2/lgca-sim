"""Run checker regressions without modifying checker sources."""
import subprocess
import sys
from pathlib import Path
from sim.cert.tables import provenance
from .protocol import write_json

def qualify(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    tests=['test_portable_verdicts','test_validation','test_checker_mutant_is_killed','test_v3_replay_regression','test_source_mutants','test_oracles_and_corruption']
    result=subprocess.run([sys.executable,'-m','unittest',*[f'sim.cert.test_cert.CertificateTests.{name}' for name in tests],'-v'],capture_output=True,text=True,timeout=600)
    log=result.stdout+result.stderr
    (out/'checker_tests.log').write_text(log,encoding='utf-8')
    # Qualification requires actual source-mutation coverage, not merely a
    # successful import or a PASS from an oracle table.
    mutation_passed=result.returncode==0 and 'CHECKER_SOURCE_MUTANTS_KILLED' in log
    record=dict(status='qualified' if mutation_passed else 'certificate-blocked',
                source_sha256=provenance()['checker_sha256'],mutation_tests_passed=mutation_passed,
                test_exit_code=result.returncode)
    write_json(out/'checker_qualification.json',record)
    if not mutation_passed: raise ValueError('checker qualification blocked; inspect checker_tests.log')
    return record
