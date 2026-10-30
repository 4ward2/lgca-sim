import json
import shlex
import unittest
from pathlib import Path
from .jobs import jobs

class JobsTests(unittest.TestCase):
    def test_artifact_paths_and_seeds(self):
        plan=jobs(); names={j['name'] for j in plan['jobs']}
        self.assertEqual(len(names),8)
        for j in plan['jobs']:
            self.assertTrue(set(j['needs'])<=names)
            self.assertIn('--output out/'+j['name']+'/',j['command'])
            self.assertTrue(shlex.split(j['command']))
        seeds=[int(j['command'].split('--seed ')[1].split()[0]) for j in plan['jobs'] if '--seed ' in j['command']]
        self.assertTrue(all(s>=60000 for s in seeds))
        self.assertTrue({60101,60102,60103}<=set(seeds))
        self.assertEqual(json.loads(Path(__file__).with_name('jobs_gate.json').read_text(encoding='utf-8-sig')),plan)
