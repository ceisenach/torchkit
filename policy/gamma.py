# Imports
import torch
from torch.autograd import Variable
from torch.distributions.gamma import Gamma as TGamma
import math
import numpy as np

import autodiff as ad
import utils as ut
from .exponential_family import ExponentialFamily2P


__all__ = ['Gamma']


class Gamma(ExponentialFamily2P):
    """
    Gamma Policy.
    """
    def __init__(self,acnet,**kwargs):
        super(Gamma,self).__init__(acnet,**kwargs)

    def kl_divergence(self,states,alpha0,beta0):
        """
        KL(pi_old || pi_new). alpha0, beta0 == pi_old
        """
        alpha1, beta1 = self._net(states.detach())
        I = torch.sum(alpha1*(torch.log(beta0) - torch.log(beta1)),dim=1,keepdim=True)
        II = torch.sum((alpha0-alpha1)*torch.polygamma(0,alpha0),dim=1,keepdim=True)
        III = torch.sum((beta1-beta0)*(alpha0/beta0),dim=1,keepdim=True)
        IV = torch.sum(torch.lgamma(alpha1) - torch.lgamma(alpha0),dim=1,keepdim=True)
        kl = I+II+III+IV
        return kl

    def log_likelihood(self,a_t_hat,s_t):
        alpha, beta = self._net(s_t.detach())
        log_prob = alpha*torch.log(beta) + (alpha-1.)*torch.log(a_t_hat) - beta*a_t_hat - torch.lgamma(alpha)
        log_prob = log_prob.sum(dim=1)
        assert log_prob.dim() == 1
        return log_prob

    def sample(self,s_t,deterministic=False,**kwargs):
        if deterministic:
            raise RuntimeError('Not supported')
        alpha,beta = self._net(s_t)
        gs = TGamma(alpha,beta)
        return gs.sample().detach()

    def fisher_information_params(self,alpha,beta,backend='pytorch'):
        I_11 = torch.polygamma(1,alpha.data)
        I_12 = -1./beta.data
        I_22 = alpha*(I_12**2)

        if backend == 'pytorch':
            pass
        elif backend == 'numpy':        
            I_11 = I_11.numpy()
            I_12 = I_12.numpy()
            I_22 = I_22.numpy()
        else:
            raise RuntimeError()

        return I_11,I_12,I_22