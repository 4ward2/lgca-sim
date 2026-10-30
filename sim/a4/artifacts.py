"""Verify train-only artifacts before exposing arrays to any model."""
from pathlib import Path
import numpy as np
from .protocol import read_json, sha

def load_training_artifact(root):
    root = Path(root)
    forbidden = [p for p in root.rglob('*') if p.is_file() and
                 (p.suffix in ('.pth', '.gz') or p.name.endswith('_data.pth') or
                  p.name.startswith('test_') and p.suffix in ('.npz','.npy') or p.name in ('inputs.npy', 'solutions.npy'))]
    if forbidden:
        raise ValueError('training input contains raw/test artifacts')
    candidates = list(root.rglob('train_dev_manifest.json'))
    if len(candidates) != 1:
        raise ValueError('expected exactly one train/dev manifest')
    manifest_path = candidates[0]
    manifest = read_json(manifest_path)
    if set(manifest['files']) != {'train.npz', 'dev.npz'}:
        raise ValueError('unexpected training artifact files')
    result = []
    for name, count in [('train',8000), ('dev',2000)]:
        path = manifest_path.parent/(name+'.npz')
        if sha(path) != manifest['files'][path.name]:
            raise ValueError('training artifact hash mismatch')
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        if data['x'].shape != (count,1,32) or data['y'].shape != (count,32):
            raise ValueError('training artifact shape')
        if len(set(data['sample_id'].tolist())) != count:
            raise ValueError('duplicate training IDs')
        if not np.isin(data['x'],[0,1]).all() or not np.array_equal(data['x'][:,0].cumsum(1)%2,data['y']):
            raise ValueError('training labels')
        result.append(data)
    train, dev = result
    if set(train['sample_id']) & set(dev['sample_id']):
        raise ValueError('split IDs overlap')
    expected = sorted(train['sample_id'].tolist())[:128]
    if manifest['calibration_ids'] != expected:
        raise ValueError('calibration IDs are not the registered training subset')
    return train, dev
