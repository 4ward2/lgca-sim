import copy
import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from sim.logic import to_internal
from sim.cert.tables import extract,interpret
from sim.a4.models import oracle,build,UnsharedBPNet
from sim.a4.evaluate import run_ac_to_fp,run_bp,run_dt_streaming
from sim.a4.metrics import score_exact,aggregate_seeds
from sim.a4.data import make_official_split,normalize_roundtrip,load_prefix
from sim.a4.ablations import flip_unknown,randomize_equal_density,deploy_variants
from sim.a4.certify import scalar_replay,certify_prefix,replay
from sim.a4.wave_cert import wave_certificate,replay as replay_wave
from sim.a4.protocol import PILOTS

class Core(unittest.TestCase):
    def test_exhaustive_oracle(self):
        model=oracle(); table=extract(model)
        for n in range(1,9):
            x=np.array(list(itertools.product((0,1),repeat=n)),dtype=np.float32)
            y=x.cumsum(1)%2; r=run_ac_to_fp(model,to_internal(x[:,None]))
            np.testing.assert_array_equal(r['pred'],y)
            self.assertLessEqual(r['t_fp'].max(),2*n+1)
            for bits,target in zip(x,y): self.assertEqual(scalar_replay(table,(bits+2).astype(int).tolist()),target.tolist())

    def test_split_roundtrip(self):
        a,b=make_official_split(); split=torch.utils.data.random_split(range(10000),[8000,2000],generator=torch.Generator().manual_seed(42))
        self.assertEqual(a.tolist(),split[0].indices); self.assertEqual(b.tolist(),split[1].indices)
        np.testing.assert_equal(normalize_roundtrip([-.5,.5]),[0,1])
        with self.assertRaises(ValueError): load_prefix('.',512,role='train')

    def test_random_table_differential(self):
        rng=np.random.default_rng(740000)
        for _ in range(8):
            m=oracle()
            for name in ('u','r','o'): getattr(m,name).data.copy_(torch.tensor(rng.integers(0,2,size=getattr(m,name).shape)))
            x=np.array(list(itertools.product((0,1),repeat=3)),dtype=np.float32)
            result=run_ac_to_fp(m,to_internal(x[:,None]),trace=True)
            table=extract(m)
            for bits,pred in zip(x,result['pred']):
                word=(bits+2).astype(int).tolist()
                self.assertEqual(scalar_replay(table,word),pred.tolist())
                self.assertEqual(interpret(table,np.array(word))[1].tolist(),pred.tolist())

    def test_output_plateau_not_fixed_point(self):
        m=oracle(); m.o.data.zero_(); x=np.ones((1,1,8),np.float32)
        result=run_ac_to_fp(m,to_internal(x),trace=True)
        self.assertGreater(result['t_fp'][0],2)
        self.assertFalse(torch.equal(result['trajectory'][1],result['trajectory'][2]))

    def test_quantization_and_variants(self):
        m=build('b',PILOTS[0]); q=copy.deepcopy(m); q.infer='max'
        np.testing.assert_array_equal(q.q(torch.tensor([-.125,.125,99.,-99.]),True).numpy(),[0,0,7.75,-7.75])
        x=np.zeros((2,1,3),np.float32); xv=to_internal(x)
        np.testing.assert_allclose(run_bp(q,xv,3,True)['pred'],q(xv,3,True)[0].detach().numpy())
        self.assertEqual(sum(p.numel() for p in m.parameters()),80)

    def test_unshared_storage_and_gates(self):
        m=UnsharedBPNet(PILOTS[0]); self.assertEqual(len(m.layers),64)
        self.assertEqual(len({p.U.data_ptr() for p in m.layers}),64)
        self.assertTrue(torch.equal(m.layers[0].U,m.layers[1].U))
        for arm in ('r','f'): self.assertEqual(build(arm,PILOTS[0]).gates_per_position(),384)
        from sim.a4.evaluate import run_depth_bp
        x=to_internal(np.zeros((2,1,3),np.float32))
        with torch.no_grad(): expected=m(x,64)[0].numpy()
        result=run_depth_bp(m,x,64)
        np.testing.assert_allclose(result['pred'],expected)
        self.assertIn(32,result['curve']); self.assertIn(64,result['curve'])

    def test_density_unknown(self):
        m=oracle(); x=np.array([[[0,1,0]],[[1,0,1]]],np.float32)
        train=dict(x=x,y=x[:,0].cumsum(1)%2,mask=np.ones((2,3),bool),diff=np.full(2,3),sample_id=np.arange(2))
        a=randomize_equal_density(m,740000,train); b=randomize_equal_density(m,740000,train)
        for name,width in [('u',2),('r',4)]:
            self.assertTrue(torch.equal(getattr(a,name).reshape(-1,width).sum(1),getattr(m,name).reshape(-1,width).sum(1)))
            self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)))
        m.r.data.fill_(1); m.o.data.zero_(); f=flip_unknown(m)
        aa=run_ac_to_fp(m,to_internal(x)); bb=run_ac_to_fp(f,to_internal(x))
        self.assertEqual(aa['state_hashes'],bb['state_hashes']); np.testing.assert_array_equal(aa['t_fp'],bb['t_fp'])
        self.assertTrue((aa['pred']!=bb['pred']).all())

    def test_certificates(self):
        from sim.b_arm import TernaryNet
        with self.assertRaisesRegex(ValueError,'V != 2'):
            certify_prefix(extract(TernaryNet(1,1,V=3)))
        t=extract(oracle()); result=certify_prefix(t); self.assertEqual(result['verdict'],'PASS'); replay(t,result)
        forged=copy.deepcopy(result); forged.pop('left_states')
        with self.assertRaises(ValueError): replay(t,forged)
        broken=copy.deepcopy(t); broken.pop('table_sha256'); broken['r'][0][0][3]=[[1,0],[1,0]]
        self.assertEqual(certify_prefix(broken)['verdict'],'FAIL')
        self.assertEqual(certify_prefix(t,timeout=0)['verdict'],'UNKNOWN')
        proof=wave_certificate(); replay_wave(proof); proof['obligations'].pop()
        with self.assertRaises(ValueError): replay_wave(proof)

    def test_exact(self):
        self.assertEqual(score_exact([[1,0,0]],[[1,1,0]])['exact'],0)
        with self.assertRaises(ValueError): aggregate_seeds([])

    def test_logic_streaming_matches_forward(self):
        from sim.a4.evaluate import run_logic
        xv=to_internal(np.array([[[0,1,0,1]]],dtype=np.float32))
        for arm in ('r','f'):
            model=build(arm,PILOTS[0]); model.eval()
            with torch.no_grad(): expected=model(xv,9,mode='hard')[0].numpy()
            result=run_logic(model,xv,9)
            np.testing.assert_array_equal(result['pred'],expected)
            self.assertIn(9,result['curve'])

    def test_dt_streaming(self):
        from sim.a4.dt_adapter import build_dt
        m=build_dt(PILOTS[0]); x=torch.tensor([[[-.5,.5,-.5]]]); m.eval()
        with torch.no_grad(): expected=m(x,iters_to_do=4)[:,-1].argmax(1).numpy()
        np.testing.assert_equal(run_dt_streaming(m,x,4)['pred'],expected)
        with torch.no_grad():
            outputs=m(x,iters_to_do=4)
            confidence=outputs.softmax(2).amax(2).sum(2)
            chosen=confidence.cummax(1).indices[:,-1]
            expected_conf=outputs[torch.arange(len(x)),chosen].argmax(1).numpy()
        np.testing.assert_equal(run_dt_streaming(m,x,4,max_conf=True)['pred'],expected_conf)

if __name__=='__main__': unittest.main()
