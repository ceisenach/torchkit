import torch
import torch.nn.functional as TF
import numpy as np
import logging
from enum import Enum
logger = logging.getLogger(__name__)

from .core import JTensor, JParameter

__all__ = ['tanh','softplus','exp','expand']

class Backend(Enum):
    PYTORCH = 1
    NUMPY = 2

##### DERIVATIVES OF COMPONENTWISE FUNCTIONS #######

# expand diagonal to matrix for derivative computation
def numpy_diag_matrix_expand(x,out_grad):
    x_expanded = np.expand_dims(x,axis=1)
    x_expanded = np.tile(x_expanded,(1,out_grad.shape[1],1))
    assert x_expanded.shape == out_grad.shape, "shape needs to be same"
    return x_expanded

def torch_diag_matrix_expand(x,out_grad):
    x_expanded = x.unsqueeze(1).repeat(1,out_grad.shape[1],1)
    assert x_expanded.shape == out_grad.shape, "shape needs to be same"
    return x_expanded

def compentwise_derivative(in_data,out_grad,torch_deriv,numpy_deriv):
    # select methods based on backend mode, prepare in_data
    backend = Backend.PYTORCH if isinstance(out_grad,torch.Tensor) else Backend.NUMPY
    dexpand = torch_diag_matrix_expand if backend == backend.PYTORCH else numpy_diag_matrix_expand
    deriv = torch_deriv if backend == backend.PYTORCH else numpy_deriv

    if backend == Backend.PYTORCH:
        x = in_data.data if isinstance(in_data,JTensor) else in_data
        assert x.dim() == 2, "Wrong dimensions"
        x,out_grad = x.detach(),out_grad.detach()
        x.requires_grad_(False), out_grad.requires_grad_(False)
    else:
        x = in_data.ndata if isinstance(in_data,JTensor) else in_data.numpy()
        assert x.ndim == 2, "Wrong dimensions"

    # compute derivative
    dd = deriv(x)
    in_grad = out_grad * dexpand(dd,out_grad)
    return in_grad

class JComponentwiseVectorFunction(object):
    @classmethod
    def _compute_jacobian(cls,out_grad,input,mode):

        in_grad = compentwise_derivative(input,out_grad,cls._torch_deriv,cls._numpy_deriv)

        # continue down graph
        if isinstance(input,JTensor):
            input.differentiate(in_grad,mode)
        else:
            logger.debug('Compute graph leaf')


#########################################################
## Tanh
#########################################################
class Jtanh(JComponentwiseVectorFunction):
    @staticmethod
    def _numpy_deriv(x):
        return 1. - np.tanh(x)**2
    @staticmethod
    def _torch_deriv(x):
        return 1. - (x.tanh())**2

def tanh(input,save_for_jacobian=False):
    x = input.data if isinstance(input,JTensor) else input
    if not save_for_jacobian:
        return x.tanh()

    out = x.tanh()
    return JTensor(out,Jtanh,input)


#########################################################
## SOFTPLUS
#########################################################
class Jsoftplus(JComponentwiseVectorFunction):
    @staticmethod
    def _numpy_deriv(x):
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)
    @staticmethod
    def _torch_deriv(x):
        exp_x = x.exp()
        return exp_x / (1.0 + exp_x)


def softplus(input,save_for_jacobian=False,alpha=1.0):
    x = input.data if isinstance(input,JTensor) else input
    if not save_for_jacobian:
        return alpha + TF.softplus(x)

    out = alpha + TF.softplus(x)
    return JTensor(out,Jsoftplus,input)


#########################################################
## Exponential
#########################################################
class Jexponential(JComponentwiseVectorFunction):
    @staticmethod
    def _numpy_deriv(x):
        return np.exp(x)
    @staticmethod
    def _torch_deriv(x):
        return x.exp()

def exp(input,save_for_jacobian=False):
    x = input.data if isinstance(input,JTensor) else input
    if not save_for_jacobian:
        return torch.exp(x)
    return JTensor(torch.exp(x),Jexponential,input)


#########################################################
## Expand
#########################################################
class JExpand(object):
    @staticmethod
    def _compute_jacobian(out_grad,input,mode):
        in_grad = out_grad
        # continue down graph
        if isinstance(input,JTensor) or isinstance(input,JParameter):
            input.differentiate(in_grad,mode)
        else:
            logger.debug('Compute graph leaf')


def expand(input,dims,save_for_jacobian=False):
    x = input.data if isinstance(input,JTensor) else input
    x = x.expand(dims)
    return x if not save_for_jacobian else JTensor(x,JExpand,input)
