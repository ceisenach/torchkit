import torch
import torch.nn as nn
import numpy as np

import autodiff as ad
from . import ExtendedTestCase

class SimpleNet(nn.Module):
    def __init__(self, sizes = [2,2,2]):
        super(SimpleNet, self).__init__()
        self.affine1 = ad.modules.Linear(sizes[0],sizes[1])
        self.affine2 = ad.modules.Linear(sizes[1],sizes[2])

    def forward(self,x,save_for_jacobian=False):
        x = self.affine1(x,save_for_jacobian)
        x = ad.functional.tanh(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = ad.functional.tanh(x,save_for_jacobian)
        return x

class DeepNet(nn.Module):
    def __init__(self, sizes = [2,2,2,2,2]):
        super(DeepNet, self).__init__()
        self.affine1 = ad.modules.Linear(sizes[0],sizes[1])
        self.affine2 = ad.modules.Linear(sizes[1],sizes[2])
        self.affine3 = ad.modules.Linear(sizes[2],sizes[3])
        self.affine4 = ad.modules.Linear(sizes[3],sizes[4])

    def forward(self,x,save_for_jacobian=False):
        x = self.affine1(x,save_for_jacobian)
        x = ad.functional.tanh(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = ad.functional.tanh(x,save_for_jacobian)
        x = self.affine3(x,save_for_jacobian)
        x = ad.functional.tanh(x,save_for_jacobian)
        x = self.affine4(x,save_for_jacobian)
        x = ad.functional.tanh(x,save_for_jacobian)
        return x


class TestAutoDiff(ExtendedTestCase):

    def test_autodiff1(self):
        net = SimpleNet()
        data = torch.ones(4,2)
        g,j = self._get_grad_jacobian(net,2,data,torch.sum)
        g2 = j.sum(dim=0,keepdim=False)
        self.assertTensorClose(g,g2)

    def test_autodiff2(self):
        net = SimpleNet([10,32,8])
        data = torch.randn(7,10)
        g,j = self._get_grad_jacobian(net,8,data,torch.sum)
        g2 = j.sum(dim=0,keepdim=False)
        self.assertTensorClose(g,g2)

    def test_autodiff3(self):
        net = SimpleNet([10,32,8])
        data = torch.randn(7,10)
        j_g,j = self._get_both_jacobians(net,8,data)
        self.assertTensorClose(j_g.view(-1),j.view(-1))

    def test_autodiff4(self):
        net = DeepNet([10,16,14,25,8])
        data = torch.randn(7,10)
        j_g,j = self._get_both_jacobians(net,8,data)
        self.assertTensorClose(j_g.view(-1),j.view(-1))


    def _get_both_jacobians(self,net,out_dim,data):
        grads = []

        ad.util.zero_jacobian(net.parameters(),out_dim)
        net_out = net(data,save_for_jacobian=True)
        
        # Jacobian
        net_out.jacobian()
        j = ad.util.gather_jacobian(net.parameters())

        # Grads
        for i in range(out_dim):
            ad.util.zero_grad(net.parameters())
            net_out = net(data)
            loss = net_out[:,i].sum()
            loss.backward()
            g = ad.util.gather_grad(net.parameters())
            grads.append(g.view(1,-1))

        j_g = torch.cat(grads,dim=0)
        return j_g,j


    def _get_grad_jacobian(self,net,out_dim,data,loss_fn):
        ad.util.zero_jacobian(net.parameters(),out_dim)
        ad.util.zero_grad(net.parameters())
        net_out = net(data,save_for_jacobian=True)
        loss = loss_fn(net_out.data)
  
        # Jacobian
        net_out.jacobian()
        j = ad.util.gather_jacobian(net.parameters())

        # Grad
        loss.backward()
        g = ad.util.gather_grad(net.parameters())

        return g,j