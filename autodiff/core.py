import torch
import torch.nn as nn

from . import util as util

__all__ = ['JParameter','JTensor']

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
        self._jacobian = None
        self.__jacobian_ready = False
        self._size_flat = util.flat_dim(self.data.shape)

    @property
    def size_flat(self):
        return self._size_flat
    

    def __repr__(self):
        return 'JParameter containing:\n' + super(JParameter, self).__repr__()

    def zero_jacobian_(self):
        self.__jacobian_ready = False
        if self._jacobian is not None:
            self._jacobian.zero_()

    def update_jacobian_(self,D,mode):
        # check if its already been updated at least once
        if not self.__jacobian_ready:
            shape = D.shape if mode=='batch' else D.shape[1:] if mode=='sum' else None
            self._jacobian = torch.zeros(shape)

        # update
        if mode == 'batch':
            pass
        elif mode == 'sum':
            D = torch.sum(D,dim=0)
        else:
            raise RuntimeError('Undefined Behavior')

        self._jacobian += D
        self.__jacobian_ready = True


    @property
    def jacobian_ready(self):
        return self.__jacobian_ready
    
    @property
    def jacobian(self):
        return self._jacobian
    


class JTensor(object):
    def __init__(self,data,creator,jacobian_info):
        self.data = data
        self._jacobian_info = jacobian_info # should contain creator (JModule or JFunc), an args and a kwargs?
        self._creator = creator

    def __repr__(self):
        return 'JTensor containing:\n' + self.data.__repr__()

    def jacobian(self,in_grad = None,mode='sum'):
        if in_grad is None:
            N = self.data.shape[0]
            d = self.data.shape[1]
            in_grad = torch.eye(d).unsqueeze(0).repeat(N,1,1)
        self._creator._compute_jacobian(in_grad,self._jacobian_info,mode)
