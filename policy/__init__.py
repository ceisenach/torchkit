import torch

class BasePolicy(object):
    """
    Base policy class, supports loading and unloading actor network
    """
    def __init__(self,acnet,**kwargs):
        self._net = acnet

    def save_model(self,path):
        sga_sd = self._net.state_dict()
        torch.save(sga_sd,path)

    def kl_divergence(self,*args,**kwargs):
        raise NotImplementedError()

    def log_likelihood(self,*args,**kwargs):
        raise NotImplementedError()

    def sample(self,*args,**kwargs):
        raise NotImplementedError()

    def load_model(self,path):
        sd = torch.load(path)
        self._net.load_state_dict(sd)

    def parameters(self,**kwargs):
        return self._net.parameters(**kwargs)

    def scale(self,action,high,low):
        return action

    def fisher_information(self,states,batch_approx=False,backend='pytorch'):
        raise NotImplementedError()

    def fisher_vector_product(self,*args,**kwargs):
        raise NotImplementedError()

    @property
    def net(self):
        return self._net




# Import important names
from .gaussian import *
from .beta import *
from .gamma import *
from .angular import *
