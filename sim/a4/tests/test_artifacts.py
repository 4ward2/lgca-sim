import tempfile
import unittest
from pathlib import Path
import numpy as np
from sim.a4.artifacts import load_training_artifact
from sim.a4.protocol import write_json,sha

class TrainingIsolation(unittest.TestCase):
    def fixture(self,root):
        for name,start,count in [('train',0,8000),('dev',8000,2000)]:
            x=np.zeros((count,1,32),np.float32)
            np.savez(root/(name+'.npz'),x=x,y=x[:,0].astype(int),sample_id=np.arange(start,start+count))
        write_json(root/'train_dev_manifest.json',dict(files={n+'.npz':sha(root/(n+'.npz')) for n in ('train','dev')},calibration_ids=list(range(128))))

    def test_valid_then_modified_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); self.fixture(root)
            train,dev=load_training_artifact(root)
            self.assertEqual(len(train['y']),8000); self.assertEqual(len(dev['y']),2000)
            with (root/'train.npz').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'): load_training_artifact(root)

    def test_raw_archive_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'prefix_sums_data.tar.gz').write_bytes(b'not opened')
            with self.assertRaisesRegex(ValueError,'raw/test'): load_training_artifact(root)

    def test_calibration_outside_train_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); self.fixture(root)
            write_json(root/'train_dev_manifest.json',dict(files={n+'.npz':sha(root/(n+'.npz')) for n in ('train','dev')},calibration_ids=list(range(8000,8128))))
            with self.assertRaisesRegex(ValueError,'calibration'): load_training_artifact(root)
