import torch


##########################################################
### Utility Functions
##########################################################
def zero_jacobian(param_list):
    for p in param_list:
        p.zero_jacobian_()

def zero_grad(param_list):
    for p in param_list:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def gather_jacobian(param_list):
    pjs = []
    for p in param_list:
        pj = p.jacobian
        if pj is not None:
            pjs.append(pj)

    jacobian = torch.cat(pjs,dim=-1)
    return jacobian


def gather_grad(param_list):
    pgs = []
    for p in param_list:
        pg = p.grad
        if pg is not None:
            pgs.append(pg.view(-1))

    grad = torch.cat(pgs,dim=0)
    return grad


def flat_dim(shape):
    fd = 1
    for di in list(shape):
        fd *= di
    return fd


# computes batch A \otimes B
# A is N x n x m matrix, B is N x p x q
# A \otimes B: N x np x mq
def bkron(A,B):
    N,n,m = A.shape
    _,p,q = B.shape

    B_tiled = B.repeat(1,n,m) # N x np x mq
    A_expanded = A.unsqueeze(2).repeat(1,1,p,1).view(N,n*p,m) # N x np x m
    A_expanded2 = A_expanded.unsqueeze(3).repeat(1,1,1,q).view(N,n*p,m*q) # N x np x mq

    assert B_tiled.size() == A_expanded2.size()

    return B_tiled * A_expanded2