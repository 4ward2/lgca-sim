"""Checkpoint serialization checks without performing local training."""
import random
import tempfile
import unittest
from pathlib import Path
import torch
from sim.a4.models import build
from sim.a4.protocol import PILOTS
from sim.a4.train import save_checkpoint,require_remote

class Checkpoint(unittest.TestCase):
    def test_rng_and_parameter_roundtrip_without_training(self):
        model=build('b',PILOTS[0]); optimizer=torch.optim.Adam(model.parameters(),lr=.03)
        generator=torch.Generator().manual_seed(PILOTS[0]); rng=random.Random(PILOTS[0])
        torch.randint(8000,(128,),generator=generator); rng.randrange(32,65)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'last.pt'
            save_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=generator.get_state(),rng=rng.getstate()))
            expected=torch.randint(8000,(128,),generator=generator); expected_t=rng.randrange(32,65)
            restored=torch.load(path,weights_only=False,map_location='cpu')
            other=build('b',PILOTS[1]); other.load_state_dict(restored['model'])
            other_optimizer=torch.optim.Adam(other.parameters(),lr=.03); other_optimizer.load_state_dict(restored['optimizer'])
            generator.set_state(restored['sampler']); rng.setstate(restored['rng'])
            self.assertTrue(torch.equal(expected,torch.randint(8000,(128,),generator=generator)))
            self.assertEqual(expected_t,rng.randrange(32,65))
            for a,b in zip(model.parameters(),other.parameters()): self.assertTrue(torch.equal(a,b))
            self.assertEqual(optimizer.state_dict(),other_optimizer.state_dict())

    def test_local_training_guard(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,{'GITHUB_ACTIONS':'false'}):
            with self.assertRaises(RuntimeError): require_remote()

    def test_resume_qualification_is_remote_only(self):
        import os
        from unittest.mock import patch
        from sim.a4.resume_check import verify
        with patch.dict(os.environ,{'GITHUB_ACTIONS':'false'}):
            with self.assertRaises(RuntimeError): verify('.')
