import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
logger = logging.getLogger(__name__)

from . import JTensor, JParameter



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

