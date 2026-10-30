import copy
import unittest
from sim.a4.report import validate_records
from sim.a4.metrics import paired_bootstrap,holm,score_exact

class Reporting(unittest.TestCase):
    def record(self):
        return dict(arm='b',seed=173101,length=40,status='complete',sample_id=list(range(10000)),exact=0.,bitacc=0.,config_sha256='cfg',split_sha256='split')

    def test_complete_and_zero(self):
        r=self.record()
        self.assertEqual(validate_records([('b',173101,40)],[r])['headline'],'complete')

    def test_reviewed_freeze_identity(self):
        import tempfile
        from pathlib import Path
        from sim.a4.report import report
        from sim.a4.protocol import write_json
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'input'; root.mkdir()
            write_json(root/'result.json',self.record())
            result=report(root,Path(temporary)/'output',[('b',173101,40)],expected_config_sha256='reviewed')
            self.assertIn('result config differs from reviewed freeze',result['issues'])

    def test_cli_failure_log(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from sim.a4.cli import main
        with tempfile.TemporaryDirectory() as temporary:
            with patch('sim.a4.cli.execute',side_effect=ValueError('fixture failure')):
                self.assertEqual(main(['smoke','--input',temporary,'--out',temporary,'--threads','2']),1)
            self.assertIn('fixture failure',(Path(temporary)/'run.log').read_text())

    def test_missing_duplicate_and_malformed(self):
        plan=[('b',173101,40)]; r=self.record()
        for records in ([],[r,r],[None],[[]]):
            self.assertEqual(validate_records(plan,records)['headline'],'protocol failure')
        for field,value in [('exact',float('nan')),('bitacc',float('inf')),('status','timeout'),('sample_id',[0]),('config_sha256',None)]:
            changed=copy.deepcopy(r); changed[field]=value
            self.assertEqual(validate_records(plan,[changed])['headline'],'protocol failure')

    def test_split_identity(self):
        a=self.record(); b=copy.deepcopy(a); b['arm']='r'; b['split_sha256']='other'
        self.assertEqual(validate_records([('b',173101,40),('r',173101,40)],[a,b])['headline'],'protocol failure')

    def test_statistics(self):
        result=paired_bootstrap([0.]*10,[1.]*10)
        self.assertEqual(result['ci95'],[1.,1.]); self.assertEqual(result['difference'],1.)
        self.assertEqual(holm([.01,.04]),[.02,.04])
        with self.assertRaises(ValueError): score_exact([[float('nan')]],[[0]])

    def test_missing_package_writes_reviewable_outputs(self):
        import tempfile
        from pathlib import Path
        from sim.a4.report import report
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'input'; root.mkdir()
            out=Path(temporary)/'output'
            result=report(root,out,[('b',173101,40)])
            self.assertEqual(result['headline'],'protocol failure')
            for name in ('summary.json','missing.json','provenance.json','per_seed.csv','report.md'):
                self.assertTrue((out/name).is_file(),name)

    def test_bad_json_does_not_hide_missing_records(self):
        import tempfile
        from pathlib import Path
        from sim.a4.report import report
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'input'; root.mkdir()
            (root/'result.json').write_text('{broken',encoding='utf-8')
            result=report(root,Path(temporary)/'out',[('b',173101,40)])
            self.assertEqual(result['headline'],'protocol failure')
            self.assertGreater(len(result['issues']),1)

    def test_complete_synthetic_package(self):
        import tempfile
        from pathlib import Path
        from sim.a4.report import report
        from sim.a4.protocol import write_json
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'input'; root.mkdir()
            record=self.record()
            write_json(root/'main'/'result.json',record)
            variants={'deployment':['soft','qbp','ac'],'unknown':['control','flipped'],'density':[f'random-{j}' for j in range(10)]}
            for kind,names in variants.items():
                for name in names:
                    row=dict(record,experiment=kind,variant=name)
                    write_json(root/kind/name/'result.json',row)
            write_json(root/'oracle'/'oracle.json',dict(length=40,exact=1,count=10000,sample_id=list(range(10000))))
            result=report(root,Path(temporary)/'out',[('b',173101,40)])
            self.assertEqual(result['headline'],'protocol failure')
            self.assertTrue(any('missing discrete table' in issue for issue in result['issues']))
            from sim.a4.models import oracle
            from sim.cert.tables import extract
            from sim.a4.certify import certify_prefix
            table=extract(oracle()); certificate=certify_prefix(table)
            for path in root.rglob('result.json'):
                from sim.a4.protocol import read_json
                row=read_json(path)
                discrete=row.get('experiment','main') in ('main','density','unknown') or row.get('variant')=='ac'
                if discrete:
                    write_json(path.with_name('table.json'),table)
                    write_json(path.with_name('certificate.json'),certificate)
            result=report(root,Path(temporary)/'complete',[('b',173101,40)])
            self.assertEqual(result['headline'],'complete',result['issues'])
