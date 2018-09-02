import torch
import torch.nn as nn
import copy
import numpy as np

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

    def zero_jacobian_(self,backend='pytorch'):
        self.__jacobian_ready = False
        if self._jacobian is not None:
            # self._jacobian = self._jacobian.detach()
            # self._jacobian.requires_grad_(False)
            if backend == 'pytorch':
                if isinstance(self._jacobian,torch.Tensor):
                    self._jacobian.zero_()
                    self._jacobian = self._jacobian.detach()
                else:
                    self._jacobian = None
            else:
                if isinstance(self._jacobian,np.ndarray):
                    self._jacobian.fill(0.)
                else:
                    self._jacobian = None

    def update_jacobian_(self,D,mode):
        # check if its already been updated at least once
        if not self.__jacobian_ready:
            shape = D.shape if mode=='batch' else D.shape[1:] if mode=='sum' else None
            if not (self._jacobian is not None and self._jacobian.shape == shape):
                # print('Mode: %s. D shape: %s. J shape: %s' %(mode,str(D.shape),str(shape)))
                if isinstance(D,torch.Tensor):
                    self._jacobian = torch.zeros(shape)
                    self._jacobian.requires_grad_(False)
                else:
                    self._jacobian = np.zeros(shape,dtype=np.float32)

        if isinstance(D,torch.Tensor):
            D = D.detach()
            if mode == 'batch':
                pass
            elif mode == 'sum':
                D = torch.sum(D,dim=0)
            else:
                raise RuntimeError('Undefined Behavior')
            self._jacobian.add_(D.data) # need to make sure not to tie up computation graph
        else:
            if mode == 'batch':
                pass
            elif mode == 'sum':
                D = np.sum(D,axis=0)
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

    @property
    def jacobian_numpy(self):
        if self._jacobian is not None:
            return copy.deepcopy(self._jacobian.numpy()) if isinstance(self._jacobian,torch.Tensor) else copy.deepcopy(self._jacobian)
        return None

    def parameters(self):
        return [self]

    def differentiate(self,in_grad = None,mode='sum',backend=None):
        # import pdb; pdb.set_trace()
        self.update_jacobian_(in_grad,mode)


class JTensor(object):
    def __init__(self,data,creator,jacobian_info):
        self.data = data
        self.ndata = self.data.detach().numpy()
        self._jacobian_info = jacobian_info # should contain creator (JModule or JFunc), an args and a kwargs?
        self._creator = creator

    def __repr__(self):
        return 'JTensor containing:\n' + self.data.__repr__()

    def differentiate(self,in_grad = None,mode='sum',backend=None):
        if in_grad is None:
            N = self.data.shape[0]
            d = self.data.shape[1]
            in_grad = torch.eye(d).unsqueeze(0).repeat(N,1,1)
        if backend is None:
            backend = 'pytorch' if isinstance(in_grad,torch.Tensor) else 'numpy'

        if backend == 'pytorch':
            in_grad = in_grad if isinstance(in_grad,torch.Tensor) else torch.from_numpy(in_grad)
        elif backend == 'numpy':
            in_grad = in_grad.detach().numpy() if isinstance(in_grad,torch.Tensor) else in_grad
        self._creator._compute_jacobian(in_grad,self._jacobian_info,mode)

        # WARNING
        self._jacobian_info = None
        self._creator = None
