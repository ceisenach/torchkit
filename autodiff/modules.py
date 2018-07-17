import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
logger = logging.getLogger(__name__)

from . import JTensor, JParameter
from . import util as util


class Linear(nn.Module):
    r"""See nn.Linear"""

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
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
        o,I = input.data if isinstance(input,JTensor) else input, out_grad
        # STEP 0 - do dimensionality check, process input
        N,d = I.shape[0],I.shape[1]
        assert I.shape[2] == self.out_features, "Bad out_grad"
        assert N == o.shape[0], "N dim mismatch"
        assert o.dim() == 2, "Inputs must be vectorized"
        oT_I = util.bkron(o.unsqueeze(1),I) # jacobian of output wrt W

        D_W = torch.sum(oT_I,dim=0)
        self.weight.jacobian += D_W

        # bias jacobian
        if self.bias is not None:
            D_b = torch.sum(I,dim=0)
            self.bias.jacobian += D_b

        # STEP 2 - compute in_grad call jacobian on inputs
        in_grad = torch.matmul(I,self.weight.data)
        if isinstance(input,JTensor):
            input.jacobian(in_grad)
        else:
            logger.debug('Compute graph leaf')

