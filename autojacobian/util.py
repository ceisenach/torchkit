import torch


##########################################################
### Utility Functions
##########################################################
def zero_jacobian(param_list,d):
    for p in param_list:
        p.zero_jacobian_(d)

def gather_jacobian(param_list):
    pjs = []
    for p in param_list:
        pj = p.jacobian
        pjs.append(pj)

    jacobian = torch.cat(pjs,dim=1)
    return jacobian

def gather_grad(param_list):
    pgs = []
    for p in param_list:
        pg = p.grad
        pgs.append(pg.view(-1,1))

    grad = torch.cat(pgs,dim=0)
    return grad

def flat_dim(shape):
    fd = 1
    for di in list(shape):
        fd *= di
    return fd