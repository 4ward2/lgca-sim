import copy
import unittest
from sim.a4.gates import pilot_gate,freeze_candidate
from sim.a4.protocol import ARMS,PILOTS

class Gates(unittest.TestCase):
    def inputs(self):
        timings=[dict(arm=a,seconds_per_update=.1,dev_seconds=1,evaluation_runner_hours=.01,warmup_updates=10,measured_updates=50,budget_qualified=True) for a in ARMS]
        runs=[dict(seed=s,status='complete',dev_ac_exact=1.) for s in PILOTS]
        return timings,runs,dict(wheel_hashes={'torch':'abc'}),dict(status='qualified',source_sha256='abc',mutation_tests_passed=True)

    def test_candidate_never_locks(self):
        args=self.inputs(); self.assertEqual(pilot_gate(*args)['status'],'READY_FOR_REVIEW')
        candidate=freeze_candidate(*args,files={})
        self.assertEqual(candidate['status'],'DRAFT'); self.assertFalse(candidate['launch_allowed'])

    def test_missing_duplicate_and_nan(self):
        args=list(self.inputs()); args[0]=args[0][:-1]
        self.assertEqual(pilot_gate(*args)['status'],'STOP')
        args=list(self.inputs()); args[0][0]['seconds_per_update']=float('nan')
        self.assertEqual(pilot_gate(*args)['status'],'STOP')

    def test_learnability_and_budget(self):
        args=list(self.inputs()); args[1][0]['dev_ac_exact']=0; args[1][1]['dev_ac_exact']=0
        self.assertIn('B learnability gate failed',pilot_gate(*args)['issues'])
        args=list(self.inputs()); args[0][0]['seconds_per_update']=100
        self.assertIn('required training run exceeds four hours',pilot_gate(*args)['issues'])

    def test_dt_budget_omission(self):
        args=list(self.inputs()); args[0][3]['seconds_per_update']=10
        result=pilot_gate(*args)
        self.assertEqual(result['dt_status'],'not-run-budget')
        self.assertEqual(result['status'],'READY_FOR_REVIEW')
