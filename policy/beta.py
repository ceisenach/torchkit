# Imports
import torch
from torch.autograd import Variable
from torch.distributions.beta import Beta as TBeta
import math
import numpy as np

import autodiff as ad
import utils as ut
from .exponential_family import ExponentialFamily2P


__all__ = ['Beta']


class Beta(ExponentialFamily2P):
    """
    Beta Policy.
    """
    def __init__(self,acnet,**kwargs):
        super(Beta,self).__init__(acnet,**kwargs)

    def kl_divergence(self,states,alpha0,beta0):
        """
        KL(pi_old || pi_new)
        """
        alpha1, beta1 = self._net(Variable(states))
        delta_lgamma = torch.lgamma(alpha0 + beta0) - torch.lgamma(alpha1 + beta1) - torch.lgamma(alpha0)\
                       - torch.lgamma(beta0) + torch.lgamma(alpha1) + torch.lgamma(beta1)
        tpab = torch.polygamma(0,alpha0 + beta0)
        delta_pgamma = (alpha0-alpha1)*(torch.polygamma(0,alpha0) - tpab) + (beta0-beta1)*(torch.polygamma(0,beta0) - tpab)
        kl = delta_pgamma + delta_lgamma
        return kl.sum(1, keepdim=True)

    def log_likelihood(self,a_t_hat,s_t):
        alpha, beta = self._net(s_t.detach())
        lx = torch.log(a_t_hat)
        l1mx = torch.log(1.-a_t_hat)
        lgamma = torch.lgamma(alpha+beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        log_prob = (alpha-1.0)*lx + (beta-1.)*l1mx + lgamma
        log_prob = log_prob.sum(dim=1)
        assert log_prob.dim() == 1
        return log_prob

    def sample(self,s_t,deterministic=False,**kwargs):
        if deterministic:
            raise RuntimeError('Not supported')
        alpha,beta = self._net(s_t)
        bs = TBeta(alpha,beta)
        return bs.sample().detach()

    def fisher_information_params(self,alpha,beta,backend='pytorch'):
        pgab = torch.polygamma(1,alpha.data+beta.data)
        pga = torch.polygamma(1,alpha.data)
        pgb = torch.polygamma(1,beta.data)
        I_11 = pga-pgab
        I_12 = -pgab
        I_22 = pgb-pgab

        if backend == 'pytorch':
            pass
        elif backend == 'numpy':        
            I_11 = I_11.numpy()
            I_12 = I_12.numpy()
            I_22 = I_22.numpy()
        else:
            raise RuntimeError()

        return I_11,I_12,I_22


    def scale(self,action,high,low):
        dist = high-low
        return (action -0.5 ) / (high-low)