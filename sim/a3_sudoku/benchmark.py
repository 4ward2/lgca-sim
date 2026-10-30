"""One-update timing probes only; no learned artifacts or performance claims."""
import json
import time
import torch
from .soft import CardinalityBP,batched_forward
from .graph import build_standard
from .train import supervised_loss


def main():
    torch.set_num_threads(2)
    rows=[]
    for n,steps in ((4,16),(9,64)):
        model=CardinalityBP(60103); opt=torch.optim.Adam(model.parameters(),lr=.01)
        b=int(n**.5)
        solution=[(r*b+r//b+c)%n for r in range(n) for c in range(n)]
        puzzle=[v if i%3==0 else -1 for i,v in enumerate(solution)]
        start=time.monotonic()
        _,history=batched_forward(model,build_standard(n),[puzzle]*32,steps,True)
        loss=supervised_loss(history,[solution]*32,[puzzle]*32,n)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        rows.append(dict(n=n,batch=32,steps=steps,seconds=time.monotonic()-start))
    print(json.dumps(rows))

if __name__=='__main__':
    main()
