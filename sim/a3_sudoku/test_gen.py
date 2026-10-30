import unittest
from .gen import generate
from .oracle import count_solutions, validate_solution

class GeneratorTests(unittest.TestCase):
    def test_unique_repeatable(self):
        a=generate(4,2,60031,10,'train')
        b=generate(4,2,60031,10,'train')
        self.assertEqual(a['records'],b['records'])
        self.assertEqual(len(a['records']),2)
        for r in a['records']:
            self.assertEqual(count_solutions(r['puzzle'],4),1)
            self.assertTrue(validate_solution(r['solution'],4,r['puzzle']))
    def test_budget(self):
        self.assertEqual(generate(9,1,60032,0,'dev')['status'],'budget')

    def test_dev_family_exists(self):
        a=generate(4,1,61001,2,'dev')
        self.assertEqual(len(a['records']),1)

    def test_resume_and_cli(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from .gen import main
        a=generate(4,1,60031,2,'train')
        b=generate(4,2,60031,2,'train',a)
        c=generate(4,2,60031,2,'train')
        self.assertEqual(b['records'],c['records'])
        with tempfile.TemporaryDirectory() as d:
            with patch('sys.argv',['gen','--n','9','--count','1','--seed','61072','--time-limit','3','--split','dev','--output',d]):
                main()
            from .manifest import replay_manifest
            manifest=json.loads((Path(d)/'manifest.json').read_text())
            self.assertEqual(len(replay_manifest(manifest)),1)
            self.assertEqual(sum(json.loads((Path(d)/'summary.json').read_text())['counts'].values()),1)
