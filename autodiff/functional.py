import torch
import torch.nn.functional as TF
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .core import JTensor

__all__ = ['tanh']

class Jtanh(object):
    @staticmethod
    def _compute_jacobian(out_grad,input,mode):
        in_grad = None
        if isinstance(out_grad,torch.Tensor):
            x = input.data if isinstance(input,JTensor) else input
            x,out_grad = x.detach(),out_grad.detach()
            x.requires_grad_(False), out_grad.requires_grad_(False)

            assert x.dim() == 2, "Wrong dimensions"
            sech2_x = 1. - (x.tanh())**2
            sech2_x_expanded = sech2_x.unsqueeze(1).repeat(1,out_grad.shape[1],1)
            assert sech2_x_expanded.shape == out_grad.shape, "shape needs to be same"
            in_grad = out_grad * sech2_x_expanded
            in_grad = in_grad.detach()
        
        else:
            x = input.ndata if isinstance(input,JTensor) else input.numpy()
            sech2_x = 1. - np.tanh(x)**2
            sech2_x_expanded = np.expand_dims(sech2_x,axis=1)
            sech2_x_expanded = np.tile(sech2_x_expanded,(1,out_grad.shape[1],1))
            assert sech2_x_expanded.shape == out_grad.shape, "shape needs to be same"
            in_grad = out_grad * sech2_x_expanded


        # continue down graph
        if isinstance(input,JTensor):
            input.jacobian(in_grad,mode)
        else:
            logger.debug('Compute graph leaf')



def tanh(input,save_for_jacobian=False):
    x = input.data if isinstance(input,JTensor) else input
    if not save_for_jacobian:
        return x.tanh()

    out = x.tanh()
    return JTensor(out,Jtanh,input)



class Jsoftplus(object):
    @staticmethod
    def _compute_jacobian(out_grad,input,mode):
        in_grad = None
        if isinstance(out_grad,torch.Tensor):
            x = input.data if isinstance(input,JTensor) else input
            x,out_grad = x.detach(),out_grad.detach()
            x.requires_grad_(False), out_grad.requires_grad_(False)

            assert x.dim() == 2, "Wrong dimensions"
            exp_x = x.exp()
            dsp = exp_x / (1.0 + exp_x)
            dsp_expanded = dsp.unsqueeze(1).repeat(1,out_grad.shape[1],1)
            assert dsp_expanded.shape == out_grad.shape, "shape needs to be same"
            in_grad = out_grad * dsp_expanded
            in_grad = in_grad.detach()
        
        else:
            x = input.ndata if isinstance(input,JTensor) else input.numpy()
            exp_x = np.exp(x)
            dsp = exp_x / (1.0 + exp_x)
            dsp_expanded = np.expand_dims(dsp,axis=1)
            dsp_expanded = np.tile(dsp_expanded,(1,out_grad.shape[1],1))
            assert dsp_expanded.shape == out_grad.shape, "shape needs to be same"
            in_grad = out_grad * dsp_expanded


        # continue down graph
        if isinstance(input,JTensor):
            input.jacobian(in_grad,mode)
        else:
            logger.debug('Compute graph leaf')



def softplus(input,save_for_jacobian=False,alpha=1.0):
    x = input.data if isinstance(input,JTensor) else input
    if not save_for_jacobian:
        return alpha + TF.softplus(x)

    out = alpha + TF.softplus(x)
    return JTensor(out,Jsoftplus,input)