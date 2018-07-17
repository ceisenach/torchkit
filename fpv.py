# from ikostrikov

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
logger = logging.getLogger(__name__)

import torch


##########################################################
### Utility Functions
##########################################################
def zero_jacobian(param_list,d):
    for p in param_list:
        p._zero_jacobian(d)

def gather_jacobian(param_list):
    # determine size to allocate
    pjs = []
    for p in param_list:
        pj = p.jacobian
        pjs.append(pj)

    jacobian = torch.cat(pjs,dim=1)
    return jacobian

def flat_dim(shape):
    fd = 1
    for di in list(shape):
        fd *= di
    return fd

##########################################################
### Auto-Jacobian
##########################################################
class JParameter(nn.Parameter):
    r"""Special type of Parameter to compute Jacobian"""
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self,**kwargs):
        self.jacobian = None
        self._size_flat = flat_dim(self.data.shape)

    @property
    def size_flat(self):
        return self._size_flat
    

    def __repr__(self):
        return 'JParameter containing:\n' + super(JParameter, self).__repr__()

    def _zero_jacobian(self,d):
        jacobian_shape = (d,self.size_flat)
        self.jacobian = torch.zeros(jacobian_shape)


class JTensor(object):
    def __init__(self,data,creator,jacobian_info):
        self.data = data
        self._jacobian_info = jacobian_info # should contain creator (JModule or JFunc), an args and a kwargs?
        self._creator = creator

    def __repr__(self):
        return 'JTensor containing:\n' + self.data.__repr__()

    def jacobian(self,in_grad = None):
        if in_grad is None:
            N = self.data.shape[0]
            d = self.data.shape[1]
            in_grad = torch.eye(d).unsqueeze(0).repeat(N,1,1)
        self._creator._compute_jacobian(in_grad,self._jacobian_info)


class Jtanh(object):
    @staticmethod
    def _compute_jacobian(out_grad,input):
        x = input.data if isinstance(input,JTensor) else input
        assert x.dim() == 2, "Wrong dimensions"
        sech2_x = 1. - (x.tanh())**2
        sech2_x_expanded = sech2_x.unsqueeze(1).repeat(1,out_grad.shape[1],1)
        assert sech2_x_expanded.shape == out_grad.shape, "shape needs to be same"
        in_grad = out_grad * sech2_x_expanded

        # continue down graph
        if isinstance(input,JTensor):
            input.jacobian(in_grad)
        else:
            logger.debug('Compute graph leaf')



def tanh(input,save_for_jacobian=False):
    x = input.data if isinstance(input,JTensor) else input
    if not save_for_jacobian:
        return x.tanh()

    out = x.tanh()
    return JTensor(out,Jtanh,input)


class JLinear(nn.Module):
    r"""See nn.Linear"""

    def __init__(self, in_features, out_features, bias=True):
        super(JLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = JParameter(data=torch.Tensor(out_features, in_features))
        if bias:
            self.bias = JParameter(data=torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, save_for_jacobian=False):
        d_input = input if not isinstance(input,JTensor) else input.data
        output = F.linear(d_input, self.weight, self.bias)
        if save_for_jacobian:
            output = JTensor(output,self,input)
        # if save for Jacobian
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    # Assume weights has shape L_1 x L_2, layer output is X W^T
    # Output of layer has shape N x L_1
    # input X has shape N x L_2
    # out_grad should have shape N x d x L_1
    def _compute_jacobian(self,out_grad,input):
        # STEP 0 - do dimensionality check, process input
        x = input.data if isinstance(input,JTensor) else input
        N,d = out_grad.shape[0],out_grad.shape[1]
        assert out_grad.shape[2] == self.out_features, "Bad out_grad"
        assert N == x.shape[0], "N dim mismatch"
        assert x.dim() == 2, "Inputs must be vectorized"

        # STEP 1 - do something to update jacobian of params
        # weight jacobian
        out_grad_tiled = out_grad.repeat(1,1,self.in_features) # N x d x L_1L_2
        x_expanded = x.unsqueeze(1).repeat(1,d,1) # N x d x L_2
        x_expanded2 = x_expanded.unsqueeze(3)
        x_expanded3 = x_expanded2.repeat(1,1,1,self.out_features) # N x d x L_2 x L_1
        x_expanded4 = x_expanded3.view(x_expanded3.shape[0],x_expanded3.shape[1],x_expanded3.shape[2]*x_expanded3.shape[3]) # N x d x L_2L_1

        assert out_grad_tiled.size() == x_expanded4.size()

        jacobian_weights = torch.sum((out_grad_tiled * x_expanded4),dim=0)
        self.weight.jacobian += jacobian_weights

        # bias jacobian
        if self.bias is not None:
            self.bias.jacobian += torch.sum(out_grad,dim=0)

        # STEP 2 - compute in_grad call jacobian on inputs
        in_grad = torch.matmul(out_grad,self.weight.data)
        if isinstance(input,JTensor):
            input.jacobian(in_grad)
        else:
            logger.debug('Compute graph leaf')


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.affine1 = JLinear(2,2)
        self.affine2 = JLinear(2,2)


    def forward(self,x,save_for_jacobian=False):
        x = self.affine1(x,save_for_jacobian)
        x = tanh(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = tanh(x,save_for_jacobian)
        return x

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    a = SimpleNet()
    data = torch.ones(4,2)
    out = a(data,save_for_jacobian=True)
    zero_jacobian(a.parameters(),data.shape[1])
    print(out)
    out.jacobian()
    j = gather_jacobian(a.parameters())
    print(j.shape)
    print(j)
