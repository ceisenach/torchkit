import torch
import numpy as np
import copy

##########################################################
### Utility Functions
##########################################################
def zero_jacobian(param_list,backend='pytorch'):
    for p in param_list:
        p.zero_jacobian_(backend)


def zero_grad(param_list):
    for p in param_list:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def gather_jacobian(param_list,backend='pytorch'):
    if backend == 'pytorch':
        pjs = []
        for p in param_list:
            pj = copy.deepcopy(p.jacobian)
            if pj is not None:
                pjs.append(pj.detach())

        jacobian = torch.cat(pjs,dim=-1)
        return jacobian
    else:
        pjs = []
        for p in param_list:
            # pj = copy.copy(p.jacobian)
            pj = p.jacobian
            if pj is not None:
                pj = np.copy(pj)
                pjs.append(pj)

        jacobian = np.concatenate(pjs,axis=-1)
        del pjs
        return jacobian


def flat_dim(shape):
    fd = 1
    for di in list(shape):
        fd *= di
    return fd


# computes batch A \otimes B
# A is N x n x m matrix, B is N x p x q
# A \otimes B: N x np x mq
def bkron(A,B):
    if isinstance(A,torch.Tensor):
        A,B = A.detach(), B.detach()
        A.requires_grad_(False), B.requires_grad_(False)
        N,n,m = A.shape
        _,p,q = B.shape

        B_tiled = B.repeat(1,n,m) # N x np x mq
        A_expanded = A.unsqueeze(2).repeat(1,1,p,1).view(N,n*p,m) # N x np x m
        A_expanded2 = A_expanded.unsqueeze(3).repeat(1,1,1,q).view(N,n*p,m*q) # N x np x mq

        assert B_tiled.size() == A_expanded2.size()

        return (B_tiled * A_expanded2).detach()

    # pure numpy implementation of bkron
    else:
        N,n,m = A.shape
        _,p,q = B.shape

        B_tiled = np.tile(B,(1,n,m)) # N x np x mq
        A_expanded = np.expand_dims(A,axis=2)
        A_expanded = np.tile(A_expanded,(1,1,p,1))
        A_expanded.shape = (N,n*p,m)
        A_expanded2 = np.expand_dims(A_expanded,axis=3)
        A_expanded2 = np.tile(A_expanded2,(1,1,1,q))
        A_expanded2.shape = (N,n*p,m*q)

        assert B_tiled.shape == A_expanded2.shape

        return B_tiled * A_expanded2