import tempfile
import unittest
from pathlib import Path
import numpy as np
from sim.a4.shards import save_shard,load_shard,validate_dag
from sim.a4.protocol import read_json

class Shards(unittest.TestCase):
    def test_resume_and_wrong_identity(self):
        with tempfile.TemporaryDirectory() as root:
            pred=np.zeros((3,4)); identity={'checkpoint':'abc','length':4}
            self.assertIsNone(load_shard(root,0,identity,3))
            save_shard(root,0,pred,identity)
            np.testing.assert_equal(load_shard(root,0,identity,3),pred)
            with self.assertRaises(ValueError): load_shard(root,0,{'checkpoint':'other'},3)
            with self.assertRaises(ValueError): load_shard(root,0,identity,4)

    def test_candidate_dag(self):
        plan=read_json(Path(__file__).parents[1]/'jobs_a4.json')
        self.assertEqual(validate_dag(plan),len(plan['jobs']))

    def test_omit_dt_matrix(self):
        from sim.a4.planning import materialize
        from sim.a4.protocol import ARMS
        plan=read_json(Path(__file__).parents[1]/'jobs_a4.json')
        result=materialize(plan,{'gate':{'status':'READY_FOR_REVIEW','arms':[a for a in ARMS if a!='dt-small']}})
        self.assertFalse(any(j['name'].startswith(('train-dt-small-','eval-dt-small-')) for j in result['jobs']))
        self.assertTrue(any(j['name']=='pilot-dt-small' for j in result['jobs']))
