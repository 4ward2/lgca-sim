import json
import tempfile
import time
import unittest
import torch
from .gen import generate
from .train import train,supervised_loss,select_checkpoint
from .soft import CardinalityBP,batched_forward
from .graph import build_standard

class TrainingTests(unittest.TestCase):
    def test_tiny_resume_and_timing(self):
        records=generate(4,2,60031,10,'train')['records']
        config=dict(seed=60101,steps=1,batch=1,lr=.01,eval_every=1,time_limit=40,eval_steps=1,threads=2)
        with tempfile.TemporaryDirectory() as d:
            t=time.monotonic(); a=train(config,records,records,d)
            b=train(config,records,records,d,d+'/last.pt')
            self.assertEqual(a['theta'],b['theta'])
            self.assertEqual(a['step'],1)
            self.assertFalse(select_checkpoint(None,dict(certified=False),1))
            print('TINY_TRAIN_SECONDS',round(time.monotonic()-t,4))
    def test_loss_masks_givens(self):
        r=generate(4,1,60031,10,'train')['records'][0]
        m=CardinalityBP(60102)
        _,h=batched_forward(m,build_standard(4),[r['puzzle']],2,True)
        loss=supervised_loss(h,[r['solution']],[r['puzzle']],4)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(torch.isfinite(m.theta.grad).all())

    def test_interrupted_resume_exact(self):
        from unittest.mock import patch
        from . import train as module
        records=generate(4,1,60031,2,'train')['records']
        config=dict(seed=60102,steps=2,batch=1,lr=.01,eval_every=1,time_limit=40,eval_steps=1,threads=2)
        real_evaluate=module.evaluate; calls=[0]
        def interrupted(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:
                raise RuntimeError('simulated interrupt')
            return real_evaluate(*args,**kwargs)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            expected=train(config,records,records,a)
            with patch.object(module,'evaluate',side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    train(config,records,records,b)
            actual=train(config,records,records,b,b+'/last.pt')
            self.assertEqual(expected['theta'],actual['theta'])
            self.assertEqual(expected['beta'],actual['beta'])
