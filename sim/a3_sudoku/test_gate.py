import unittest
from .gate import continue_decision

class GateTests(unittest.TestCase):
    def test_incomplete_and_failure(self):
        self.assertFalse(continue_decision([dict(certified=True,s1_agreement=True,wrong_deletions=0,search_complete=False,node_ratio=.5,net_faster=True)]*3))
        good=dict(certified=True,s1_agreement=True,wrong_deletions=0,search_complete=True,node_ratio=.8,net_faster=True)
        self.assertTrue(continue_decision([good]*3))
        self.assertFalse(continue_decision([dict(good,wrong_deletions=1)]*3))

    def test_real_audit(self):
        from .gen import generate
        from .manifest import make_manifest
        from .soft import CardinalityBP
        from .gate import audit
        import torch
        records=generate(4,1,61001,2,'dev')['records']
        records=make_manifest(dict(train=[],dev=records,test=[]),61001)['records']
        model=CardinalityBP(60101)
        with torch.no_grad():
            model.theta.copy_(torch.tensor([-3.,0.,-3.]))
        result=audit(model,records,60101,search_count=100,steps=1)
        self.assertTrue(result['certified']); self.assertTrue(result['s1_agreement'])
        self.assertEqual(result['wrong_deletions'],0)
        self.assertFalse(result['search_complete'])

    def test_cli_artifacts(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from .gen import generate
        from .manifest import make_manifest
        from .soft import CardinalityBP
        from .gate import main
        records=generate(4,1,61001,2,'dev')['records']
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=root/'manifest.json'
            manifest.write_text(json.dumps(make_manifest(dict(train=[],dev=records,test=[]),61001)))
            runs=[]
            for seed in (60101,60102,60103):
                run=root/str(seed); run.mkdir(); runs.append(str(run))
                model=CardinalityBP(seed)
                with torch.no_grad():
                    model.theta.copy_(torch.tensor([-3.,0.,-3.]))
                torch.save(model.state_dict(),run/'selected.pt')
                (run/'final.json').write_text(json.dumps(dict(seed=seed,certifiable_learning_success=True)))
            with patch('sys.argv',['gate','--dev',str(manifest),'--runs',*runs,'--steps','1','--output',str(root/'out')]):
                main()
            result=json.loads((root/'out/gate.json').read_text())
            self.assertFalse(result['continue_experiment']); self.assertEqual(len(result['results']),3)
            self.assertIn('Continue: False',(root/'out/gate.md').read_text())
