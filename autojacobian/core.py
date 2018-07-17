import torch
import torch.nn as nn

from . import util as util

##########################################################
### Auto-Jacobian
##########################################################
class JParameter(nn.Parameter):
    r"""Special type of Parameter to compute Jacobian"""
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self,**kwargs):
        self.jacobian = None
        self._size_flat = util.flat_dim(self.data.shape)

    @property
    def size_flat(self):
        return self._size_flat
    

    def __repr__(self):
        return 'JParameter containing:\n' + super(JParameter, self).__repr__()

    def zero_jacobian_(self,d):
        jacobian_shape = (d,self.size_flat)
        self.jacobian = torch.zeros(jacobian_shape)


class JTensor(object):
    def __init__(self,data,creator,jacobian_info):
        self.data = data
        self._jacobian_info = jacobian_info # should contain creator (JModule or JFunc), an args and a kwargs?
        self._creator = creator

    def __repr__(self):
        return 'JTensor containing:\n' + self.data.__repr__()

    def jacobian(self,in_grad = None):
        if in_grad is None:
            N = self.data.shape[0]
            d = self.data.shape[1]
            in_grad = torch.eye(d).unsqueeze(0).repeat(N,1,1)
        self._creator._compute_jacobian(in_grad,self._jacobian_info)
