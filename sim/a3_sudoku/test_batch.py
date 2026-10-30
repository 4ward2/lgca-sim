import unittest
import torch
from .soft import CardinalityBP,batched_forward
from .graph import build_standard

class BatchTests(unittest.TestCase):
    def test_matches_and_gradients(self):
        m=CardinalityBP(60040).double(); g=build_standard(4)
        p=[0]+[-1]*15
        a=m(g,p,2); b=batched_forward(m,g,[p],2)[0]
        torch.testing.assert_close(a,b)
        a[torch.isfinite(a)].sum().backward(); grad=m.theta.grad.clone(); m.zero_grad()
        b[torch.isfinite(b)].sum().backward()
        torch.testing.assert_close(grad,m.theta.grad)
