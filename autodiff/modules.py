import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
logger = logging.getLogger(__name__)

from .core import JTensor
from .core import JParameter as Parameter
from . import util as util

__all__ = ['Parameter','Module','Linear']

class Module(nn.Module):

    def _compute_jacobian(self,out_grad,input):
        raise NotImplementedError()


class Linear(Module):
    r"""See nn.Linear"""

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(data=torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(data=torch.Tensor(out_features))
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
    def _compute_jacobian(self,out_grad,input,mode):
        # Check which implementation to use
        in_grad,I,I_oT = None,None,None
        if isinstance(out_grad,torch.Tensor):
            o,I = input.data if isinstance(input,JTensor) else input, out_grad
            # STEP 0 - do dimensionality check, process input
            N,d = I.shape[0],I.shape[1]
            assert I.shape[2] == self.out_features, "Bad out_grad"
            assert N == o.shape[0], "N dim mismatch"
            assert o.dim() == 2, "Inputs must be vectorized"

            # STEP 1 -- compute Jacobian
            # Note - technically this should be o^T \otimes I, but python is row-major
            # see main.tex for more details.
            I_oT = util.bkron(I,o.unsqueeze(1)) # jacobian of output wrt W

            # STEP 2 - compute in_grad
            in_grad = torch.matmul(I,self.weight.data)
            in_grad = in_grad.detach()
        else:
            o,I = input.ndata if isinstance(input,JTensor) else input.numpy(), out_grad
            o = np.expand_dims(o,axis=1)
            I_oT = util.bkron(I,o)
            in_grad = np.matmul(I,self.weight.data.numpy())


        # STEP 3 - Do updates
        self.weight.update_jacobian_(I_oT,mode)
        if self.bias is not None:
            self.bias.update_jacobian_(I,mode)


        # STEP 4 - call Jacobian on inputs
        if isinstance(input,JTensor):
            print(type(in_grad))
            input.jacobian(in_grad,mode)
        else:
            logger.debug('Compute graph leaf')

