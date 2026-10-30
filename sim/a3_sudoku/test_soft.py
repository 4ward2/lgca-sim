import itertools
import unittest
import torch
from .soft import CardinalityBP, factor_messages, log_count_convolve
from .rounding import cardinality_to_relation, certify_eo
from .domains import read_solution
from .graph import build_standard


class Soft(unittest.TestCase):
    def test_brute_messages_and_gradients(self):
        for m in range(1,5):
            incoming=torch.linspace(-.8,.7,2*m,dtype=torch.double).reshape(m,2).requires_grad_()
            theta=torch.tensor([-.4,.3,-.9],dtype=torch.double,requires_grad=True)
            actual=factor_messages(incoming,theta)
            rows=[]
            for i in range(m):
                row=[]
                for h in (0,1):
                    terms=[]
                    for bits in itertools.product((0,1),repeat=m-1):
                        js=[j for j in range(m) if j!=i]
                        terms.append(theta[min(2,h+sum(bits))]+sum((incoming[j,z] for j,z in zip(js,bits)),incoming.new_zeros(())))
                    row.append(torch.logsumexp(torch.stack(terms),0))
                row=torch.stack(row)
                rows.append(row-torch.logsumexp(row,0))
            torch.testing.assert_close(actual,torch.stack(rows))
            self.assertTrue(torch.autograd.gradcheck(factor_messages,(incoming,theta),eps=1e-6,atol=1e-5))
    def test_infinities(self):
        x=torch.tensor([[0.,-torch.inf],[-torch.inf,-torch.inf]],requires_grad=True)
        theta=torch.tensor([-torch.inf,0.,-torch.inf],requires_grad=True)
        y=factor_messages(x,theta)
        self.assertFalse(torch.isnan(y).any())
        torch.where(torch.isfinite(y),y,torch.zeros_like(y)).sum().backward()
        self.assertFalse(torch.isnan(x.grad).any())
        g=build_standard(4)
        model=CardinalityBP(seed=60003).double()
        beliefs=model(g,[-1]*16,steps=2)
        self.assertEqual(tuple(beliefs.shape),(64,2))
        self.assertTrue(torch.isfinite(beliefs).all())
        model(g,[0]+[-1]*15,steps=2).exp().sum().backward()
        self.assertTrue(torch.isfinite(model.theta.grad).all())
    def test_rounding(self):
        self.assertEqual(cardinality_to_relation([-2,0,-2]),(False,True,False))
        self.assertTrue(certify_eo((False,True,False)))
        for mask in itertools.product((False,True),repeat=3):
            self.assertEqual(certify_eo(mask),mask==(False,True,False))
        g=build_standard(4)
        for d in ([0]*64,[3]*64,[1]*64):
            self.assertIsNone(read_solution(g,d))
