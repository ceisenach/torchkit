import torch
import logging
logger = logging.getLogger(__name__)

from .core import JTensor

__all__ = ['tanh']

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