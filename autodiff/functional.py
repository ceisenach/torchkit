import torch
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
            raise RuntimeError()
            # x,out_grad = torch.from_numpy(x), torch.from_numpy(out_grad)


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