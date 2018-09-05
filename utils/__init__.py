import numpy as np
import torch
import copy


def get_flat_params_from(model,backend='pytorch',ordered=False):
    """
    Get parameters from model as a single vector.
    backend signifies if a np.ndarray or torch.Tensor should be returned
    """
    params = []
    for param in model.parameters(ordered=ordered):
        if backend == 'pytorch':
            params.append(param.data.view(-1))
        else:
            params.append(param.data.view(-1).numpy())

    flat_params = torch.cat(params) if backend == 'pytorch' else np.concatenate(params,axis=0)
    return flat_params


def set_flat_params_to(model,flat_params,ordered=False):
    """
    Copy data from flat_params to parameters in model.
    flat_params can be torch.Tensor or np.ndarray
    """
    prev_ind = 0
    flat_params = flat_params if isinstance(flat_params,torch.Tensor) else torch.from_numpy(flat_params)
    for param in model.parameters(ordered=ordered):
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False,backend='pytorch'):
    """
    Get gradient from model as a single vector.
    backend signifies if a np.ndarray or torch.Tensor should be returned
    grad_grad=True signifies a second derivative.
    """
    grads = []
    for param in net.parameters():
        if backend == 'pytorch':
            if grad_grad:
                grads.append(param.grad.grad.view(-1))
            else:
                grads.append(param.grad.contiguous().view(-1))
        else:
            if grad_grad:
                grads.append(copy.copy(param.grad.grad.view(-1).numpy()))
            else:
                grads.append(copy.copy(param.grad.contiguous().view(-1).numpy()))

    flat_grad = torch.cat(grads).data if backend == 'pytorch' else np.concatenate(grads,axis=0)
    return flat_grad


def count_params(net,only_grad = False):
    """
    Count number of parameters with grad or all parameters in model.
    """
    if only_grad:
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    return sum(p.numel() for p in net.parameters())


#IMPORT NAMES
from .data import *