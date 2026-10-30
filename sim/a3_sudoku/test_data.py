import json
import unittest
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from .data import parse_pairs, family_split, augment, classify_level
from .manifest import make_manifest, replay_manifest


class Data(unittest.TestCase):
    def setUp(self):
        self.sol=tuple((r*2+r//2+c)%4 for r in range(4) for c in range(4))
        self.row={'quizzes':'.'+''.join(str(v+1) for v in self.sol[1:]),
                  'solutions':''.join(str(v+1) for v in self.sol),'family':'generated-60004'}
    def test_parsing_labels(self):
        records=parse_pairs([self.row],n=4,source='synthetic',uniqueness_seconds=.2)
        self.assertEqual(records[0]['uniqueness'],'UNIQUE')
        self.assertEqual(classify_level(self.sol,4)['level'],'S0')
        self.assertEqual(classify_level(records[0]['puzzle'],4)['level'],'S1')
        self.assertEqual(classify_level((-1,)*16,4)['level'],'MULTI')
        self.assertEqual(classify_level([0,0]+[-1]*14,4)['level'],'INVALID')
        unsat=[2,3,-1,-1,-1,-1,-1,0,1,-1,-1,3,-1,-1,-1,2]
        self.assertEqual(classify_level(unsat,4)['level'],'UNSAT')
        row=dict(self.row,quizzes='.'*16)
        self.assertEqual(parse_pairs([row],4)[0]['uniqueness'],'MULTI')
        bad=dict(self.row,solutions='1'*16)
        with self.assertRaises(ValueError):
            parse_pairs([bad],4)
        # Extreme field names and zero blanks in the CSV schema.
        extreme={'question':self.row['quizzes'].replace('.','0'),'answer':self.row['solutions']}
        self.assertEqual(parse_pairs([extreme],4)[0]['puzzle'],records[0]['puzzle'])
    def test_split_augment_manifest(self):
        records=[]
        for c in range(16):
            row=dict(self.row,quizzes=''.join('.' if j in (c,(c+1)%16) else str(v+1) for j,v in enumerate(self.sol)),family=f'f{c//2}')
            records.extend(parse_pairs([row],4,source='one'))
        records+=parse_pairs([self.row],4,source='two')
        records+=parse_pairs([self.row],4,source='three')
        split=family_split(records,seed=60005)
        owners={}
        for name,rs in split.items():
            for rec in rs:
                for family in rec['families']:
                    self.assertIn(owners.setdefault(family,name),(name,))
                child=augment(rec,seed=60006)
                self.assertEqual(child['split'],name)
                self.assertEqual(child['family'],rec['family'])
        self.assertEqual(sum(len(rs) for rs in split.values()),6)
        manifest=make_manifest(split,seed=60007,copies=2)
        restored=replay_manifest(json.loads(json.dumps(manifest)))
        self.assertEqual(restored,replay_manifest(manifest))
        self.assertEqual([r['sample_id'] for r in restored],manifest['order'])
        manifest['records'][0]['solution'][0]=99
        with self.assertRaises(ValueError):
            replay_manifest(manifest)
    def test_symmetry_dedup_and_timeout(self):
        records=parse_pairs([self.row],4)
        child=augment(dict(records[0],split='train'),60008)
        split=family_split(records+[dict(child,source='other',family='other')],60009)
        self.assertEqual(sum(map(len,split.values())),1)
        self.assertEqual(classify_level((-1,)*16,4,uniqueness_seconds=0)['level'],'UNVERIFIED')
    def test_cli_budget_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'input.jsonl').write_text(json.dumps(self.row)+'\n',encoding='utf-8')
            cmd=[sys.executable,'-m','sim.a3_sudoku.manifest','--input',str(p/'input.jsonl'),
                 '--output',str(p/'output.json'),'--n','4','--seed','60011','--max-records','1',
                 '--uniqueness-seconds','.2','--source-url','synthetic:local','--revision','v1',
                 '--license','synthetic-test']
            result=subprocess.run(cmd+['--seconds','10'],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(replay_manifest(json.loads((p/'output.json').read_text()))),1)
            original=(p/'output.json').read_bytes()
            result=subprocess.run(cmd+['--seconds','.001'],capture_output=True,text=True,timeout=15)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((p/'output.json').read_bytes(),original)
