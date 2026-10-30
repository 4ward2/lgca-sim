"""Exhaustive local orientation/membership and generator rejection checks."""
import copy,hashlib,itertools,json,tempfile,unittest
from pathlib import Path
from sim.cert import tables
from sim.rtl.gen_sv import checked,generate

class LocalTests(unittest.TestCase):
    def test_all_fixture_truth_tables(self):
        count=0
        for path in (Path(__file__).resolve().parents[1]/'cert/fixtures').glob('*_tables.json'):
            t=json.loads(path.read_text()); checked(t,path.name[0]); count+=1
            for e,a,b,d,right in itertools.product(range(t['dims']),range(1<<(t['c']+1)),range(1<<(t['c']+1)),range(4),(False,True)):
                R=t['r'][e][a][b]
                expected=sum(int(any(((d>>j)&1) and (R[j][h] if right else R[h][j]) for j in range(2)))<<h for h in range(2))
                self.assertEqual(tables.support(t,e,a,b,d,right),expected)
        self.assertEqual(count,30)
    def test_rejection(self):
        path=Path(__file__).resolve().parents[1]/'cert/fixtures/P_B_s1_tables.json'; t=json.loads(path.read_text())
        for key,value in [('V',3),('dims',2),('c',2),('u',[]),('table_sha256','bad'),('mystery',1)]:
            bad=copy.deepcopy(t); bad[key]=value
            with self.assertRaises(ValueError): checked(bad,'P')
        with tempfile.TemporaryDirectory() as d:
            for shape in ([],[0],[2,2]):
                with self.assertRaises(ValueError): generate(t,'P',shape,d)
    def test_report(self):
        from sim.rtl.report import parse
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'utilization_route.rpt').write_text('| Slice LUTs | 1,234 |\n| Slice Registers | 42 |')
            (p/'timing_route.rpt').write_text('WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns)\n------- ------- -------- -------- -------- ------\n 0.123 0.000 0 100 0.010 0.000\n')
            r=parse(d); self.assertEqual(r['resources']['Slice LUTs'],1234); self.assertEqual(r['wns_ns'],.123); self.assertEqual(r['whs_ns'],.010)
            self.assertFalse(r['verified_implementation'])
if __name__=='__main__': unittest.main()
