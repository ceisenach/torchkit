import torch
import numpy as np

import autodiff as ad
import autodiff.modules as nn
import autodiff.functional as F
import utils as ut
from . import ExtendedTestCase

class SimpleNet(nn.Module):
    def __init__(self, sizes = [2,2,2]):
        super(SimpleNet, self).__init__()
        self.affine1 = nn.Linear(sizes[0],sizes[1])
        self.affine2 = nn.Linear(sizes[1],sizes[2])

    def forward(self,x,save_for_jacobian=False):
        x = self.affine1(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
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
        x = F.tanh(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        x = self.affine3(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        x = self.affine4(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        return x

class DeepNet2(nn.Module):
    def __init__(self, sizes = [2,2,2,2,2]):
        super(DeepNet2, self).__init__()
        self.affine1 = ad.modules.Linear(sizes[0],sizes[1])
        self.affine2 = ad.modules.Linear(sizes[1],sizes[2])
        self.affine3 = ad.modules.Linear(sizes[2],sizes[3])
        self.affine4 = ad.modules.Linear(sizes[3],sizes[4])

    def forward(self,x,save_for_jacobian=False):
        x = self.affine1(x,save_for_jacobian)
        x = F.softplus(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = F.softplus(x,save_for_jacobian)
        x = self.affine3(x,save_for_jacobian)
        x = F.softplus(x,save_for_jacobian)
        x = self.affine4(x,save_for_jacobian)
        x = F.softplus(x,save_for_jacobian)
        return x

class DeepNet3(nn.Module):
    def __init__(self, sizes = [2,2,2,2,2]):
        super(DeepNet3, self).__init__()
        self.affine1 = ad.modules.Linear(sizes[0],sizes[1])
        self.affine2 = ad.modules.Linear(sizes[1],sizes[2])
        self.affine3 = ad.modules.Linear(sizes[2],sizes[3])
        self.affine4 = ad.modules.Linear(sizes[3],sizes[4])

    def forward(self,x,save_for_jacobian=False):
        x = self.affine1(x,save_for_jacobian)
        x = F.exp(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        x = self.affine3(x,save_for_jacobian)
        x = F.exp(x,save_for_jacobian)
        x = self.affine4(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
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

    def test_autodiff3(self,backend='pytorch'):
        net = SimpleNet([10,32,8])
        data = torch.randn(7,10)
        j_g,j = self._get_both_jacobians(net,8,data,backend=backend)
        self.assertTensorClose(j_g.view(-1),j.view(-1))

    def test_autodiff3_numpy(self):
        self.test_autodiff3(backend='numpy')

    def test_autodiff4(self,backend='pytorch'):
        net = DeepNet([10,16,14,25,8])
        data = torch.randn(7,10)
        j_g,j = self._get_both_jacobians(net,8,data,backend=backend)
        self.assertTensorClose(j_g.view(-1),j.view(-1))

    def test_autodiff4_numpy(self):
        self.test_autodiff4(backend='numpy')

    def test_autodiff5(self,backend='pytorch'):
        net = DeepNet2([10,16,14,25,8])
        data = torch.randn(7,10)
        j_g,j = self._get_both_jacobians(net,8,data,backend=backend)
        self.assertTensorClose(j_g.view(-1),j.view(-1))

    def test_autodiff5_numpy(self):
        self.test_autodiff5(backend='numpy')

    def test_autodiff6(self,backend='pytorch'):
        net = DeepNet3([10,16,14,25,8])
        data = torch.randn(7,10)
        j_g,j = self._get_both_jacobians(net,8,data,backend=backend)
        self.assertTensorClose(j_g.view(-1),j.view(-1))

    def test_autodiff6_numpy(self):
        self.test_autodiff6(backend='numpy')


    def test_autodiff_batch(self):
        net = DeepNet([10,16,14,25,8])
        data = torch.randn(7,10)
        j_g,j_N = self._get_both_jacobians(net,8,data,mode='batch')
        j = torch.sum(j_N,dim=0)
        self.assertTensorClose(j_g.view(-1),j.view(-1))


    def test_autodiff_batch_numpy(self):
        net = DeepNet([10,16,14,25,8])
        data = torch.randn(7,10)

        # Torch implementation
        ad.util.zero_jacobian(net.parameters())
        net_out = net(data,save_for_jacobian=True)
        net_out.differentiate(mode='batch')
        j_N_t = ad.util.gather_jacobian(net.parameters())

        # Numpy backend
        ad.util.zero_jacobian(net.parameters(),backend='numpy')
        net_out = net(data,save_for_jacobian=True)
        net_out.differentiate(mode='batch',backend='numpy')
        j_N_n = ad.util.gather_jacobian(net.parameters(),backend='numpy')

        # Compare
        j_t = torch.sum(j_N_t,dim=0)
        j_n = torch.sum(torch.from_numpy(j_N_n),dim=0)
        self.assertTensorClose(j_t.view(-1),j_n.view(-1))


    def _get_both_jacobians(self,net,out_dim,data,mode='sum',backend='pytorch'):
        grads = []
        ad.util.zero_jacobian(net.parameters(),backend=backend)
        net_out = net(data,save_for_jacobian=True)
        # Jacobian
        net_out.differentiate(mode=mode,backend=backend)
        j = ad.util.gather_jacobian(net.parameters(),backend=backend)
        j = j if backend == 'pytorch' else torch.from_numpy(j)

        # Grads
        for i in range(out_dim):
            ad.util.zero_grad(net.parameters())
            net_out = net(data)
            loss = net_out[:,i].sum()
            loss.backward()
            g = ut.get_flat_grad_from(net)
            grads.append(g.view(1,-1))

        j_g = torch.cat(grads,dim=0)
        return j_g,j


    def _get_grad_jacobian(self,net,out_dim,data,loss_fn):
        ad.util.zero_jacobian(net.parameters())
        ad.util.zero_grad(net.parameters())
        net_out = net(data,save_for_jacobian=True)
        loss = loss_fn(net_out.data)
  
        # Jacobian
        net_out.differentiate(mode='sum')
        j = ad.util.gather_jacobian(net.parameters())

        # Grad
        loss.backward()
        g = ut.get_flat_grad_from(net)

        return g,j