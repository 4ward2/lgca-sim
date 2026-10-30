"""Remote-only mathematical resume check on an explicit synthetic fixture."""
import tempfile
from pathlib import Path
import random
import numpy as np
import torch
from .train import require_remote,save_checkpoint
from .models import build,predict
from .protocol import PILOTS,write_json
from sim.logic import to_internal

def verify(out):
    require_remote()
    x=np.array([[[0,1,0,1]],[[1,0,1,0]]],dtype=np.float32)
    target=torch.tensor(x[:,0].cumsum(1)%2,dtype=torch.float32)
    def setup():
        model=build('b',PILOTS[0]); optimizer=torch.optim.Adam(model.parameters(),lr=.03)
        generator=torch.Generator().manual_seed(PILOTS[0])
        return model,optimizer,generator
    def update(model,optimizer,generator):
        ids=torch.randint(2,(2,),generator=generator); T=int(torch.randint(2,5,(1,),generator=generator))
        optimizer.zero_grad(); loss=torch.nn.functional.binary_cross_entropy(predict(model,'b',to_internal(x[ids]),T),target[ids])
        loss.backward(); optimizer.step()
        return ids.tolist(),T
    baseline,opt,g=setup(); expected=[update(baseline,opt,g) for _ in range(4)]
    split,split_opt,split_g=setup(); actual=[update(split,split_opt,split_g) for _ in range(2)]
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        path=Path(temporary)/'resume.pt'
        save_checkpoint(path,dict(model=split.state_dict(),optimizer=split_opt.state_dict(),sampler=split_g.get_state()))
        saved=torch.load(path,weights_only=False,map_location='cpu')
        restored,restored_opt,restored_g=setup(); restored.load_state_dict(saved['model'])
        restored_opt.load_state_dict(saved['optimizer']); restored_g.set_state(saved['sampler'])
        actual += [update(restored,restored_opt,restored_g) for _ in range(2)]
    if actual!=expected: raise AssertionError('resumed sample/T sequence differs')
    for a,b in zip(baseline.parameters(),restored.parameters()):
        if not torch.equal(a,b): raise AssertionError('resumed parameter trajectory differs')
    for key,values in opt.state_dict()['state'].items():
        for field,value in values.items():
            other=restored_opt.state_dict()['state'][key][field]
            if isinstance(value,torch.Tensor) and not torch.equal(value,other): raise AssertionError('optimizer trajectory differs')
    record=dict(status='pass',updates=4,interruption_after=2,seed=PILOTS[0],fixture='explicit synthetic L4; not A4 data')
    write_json(Path(out)/'resume_check.json',record)
    return record
