# from ikostrikov

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

import torch

class JParameter(nn.Parameter):
    r"""Special type of Parameter to compute Jacobian"""
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self,**kwargs):
        self.jacobian = None
        flat_dim = 1
        for di in list(self.data.shape):
            flat_dim *= di
        print(flat_dim)
        self._size_flat = flat_dim

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


def zero_jacobian(param_list,d):
    for p in param_list:
        p._zero_jacobian(d)

def gather_jacobian(param_list,d):
    pass


class JLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=False):
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
        # so far, only does it for weights
        out_grad_tiled = out_grad.repeat(1,1,self.in_features) # N x d x L_1L_2
        x_expanded = x.unsqueeze(1).repeat(1,d,1) # N x d x L_2
        x_expanded2 = x_expanded.unsqueeze(3)
        x_expanded3 = x_expanded2.repeat(1,1,1,self.out_features) # N x d x L_2 x L_1
        x_expanded4 = x_expanded3.view(x_expanded3.shape[0],x_expanded3.shape[1],x_expanded3.shape[2]*x_expanded3.shape[3]) # N x d x L_2L_1

        assert out_grad_tiled.size() == x_expanded4.size()

        jacobian_weights = torch.sum((out_grad_tiled * x_expanded4),dim=0)
        # import pdb; pdb.set_trace()
        self.weight.jacobian += jacobian_weights

        # STEP 2 - compute in_grad call jacobian on inputs
        # import pdb; pdb.set_trace()
        in_grad = torch.matmul(out_grad,self.weight.data)
        if isinstance(input,JTensor):
            input.jacobian(in_grad)
        else:
            print('Compute graph leaf')


# class Policy(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(num_inputs, 10)
#         self.affine2 = nn.Linear(10, num_outputs)
#         self.affine2.weight.data.mul_(0.1)
#         self.affine2.bias.data.mul_(0.0)
        
#         self._saved_for_jacobian = False
#         self._saved_res = None


#     def forward(self, x,save_for_jacobian=False):
#         # saved results
#         O = [] 



#         o1 = F.tanh(self.affine1(x))
#         affine2 = self.affine2(x)




#         return affine2


#     def jacobian(self):
#         if self._saved_for_jacobian:
#             # do something
#             return
#         raise RuntimeError('Save for backward not requested')


a = JLinear(10,10)
# for p in a.parameters():
#     print(p)

data = torch.ones(10,10)

out = a(data,save_for_jacobian=True)
zero_jacobian(a.parameters(),data.shape[1])
print(out)
out.jacobian()


# print(a.jacobian)
# a._zero_jacobian(4)
# print(a.jacobian)
