# Imports
import torch
from torch.autograd import Variable
from torch.distributions.beta import Beta as TBeta
import math
import numpy as np

import autodiff as ad
import utils as ut
from . import BasePolicy
from .exponential_family import ExponentialFamily2P


__all__ = ['GaussianML','GaussianMS']


class GaussianML(ExponentialFamily2P):
    """
    Gaussian Policy.

    Assumes that the network models the mean and the log standard deviation
    of a Gaussian model. The covariance is a diagonal matrix.
    """
    def __init__(self,acnet,**kwargs):
        super(GaussianML,self).__init__(acnet,**kwargs)

    def kl_divergence(self,states,mean0,log_std0):
        """
        KL(pi_old || pi_new)
        """
        std0 = torch.exp(log_std0).detach()
        mean1, log_std1 = self._net(Variable(states))
        std1 = torch.exp(log_std1)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_likelihood(self,a_t_hat,s_t):
        action_means, action_log_stds = self._net(s_t)
        action_stds = torch.exp(action_log_stds)
        var = action_stds.pow(2)
        log_prob = -(a_t_hat - action_means).pow(2) / (2 * var) - action_log_stds #- 0.5 * math.log(2 * math.pi)
        log_prob = log_prob.sum(1)
        assert log_prob.dim() == 1
        return log_prob


    def sample(self,s_t,deterministic=False,**kwargs):
        action_mean,action_log_std = self._net(s_t)
        action_std = torch.exp(action_log_std)
        if not deterministic:
            action = torch.normal(action_mean, action_std)
            return action.detach()
        return action_mean.detach()

    def fisher_information_params(self,mean,log_std_dev,backend='pytorch'):
        with torch.autograd.no_grad():
            N,d = mean.shape
            I_11 = torch.exp(-2*log_std_dev)
            I_12 = None
            I_22 = 2.0*torch.ones(N,d)

            if backend == 'pytorch':
                pass
            elif backend == 'numpy':        
                I_11 = I_11.numpy()
                I_22 = I_22.numpy()
            else:
                raise RuntimeError()

            return I_11,I_12,I_22



class GaussianMS(ExponentialFamily2P):
    """
    Gaussian Policy.

    Assumes that the network models the mean and the standard deviation
    of a Gaussian model. The covariance is a diagonal matrix.
    """
    def __init__(self,acnet,**kwargs):
        super(GaussianMS,self).__init__(acnet,**kwargs)

    def kl_divergence(self,states,mean0,std0):
        """
        KL(pi_old || pi_new)
        """
        log_std0 = torch.log(std0).detach()
        mean1, std1 = self._net(Variable(states))
        log_std1 = torch.log(std1)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_likelihood(self,a_t_hat,s_t):
        action_means, action_stds = self._net(s_t)
        action_log_stds = torch.log(action_stds)
        var = action_stds.pow(2)
        log_prob = -(a_t_hat - action_means).pow(2) / (2 * var) - action_log_stds #- 0.5 * math.log(2 * math.pi)
        log_prob = log_prob.sum(1)
        assert log_prob.dim() == 1
        return log_prob


    def sample(self,s_t,deterministic=False,**kwargs):
        with torch.autograd.no_grad():
            action_mean,action_std = self._net(s_t)
            if not deterministic:
                action = torch.normal(action_mean, action_std)
                return action
            return action_mean

    def fisher_information_params(self,mean,std_dev,backend='pytorch'):
        with torch.autograd.no_grad():
            N,d = mean.shape
            I_11 = std_dev.pow(-2)
            I_12 = None
            I_22 = 2*I_11

            if backend == 'pytorch':
                pass
            elif backend == 'numpy':        
                I_11 = I_11.numpy()
                I_22 = I_22.numpy()
            else:
                raise RuntimeError()

            return I_11,I_12,I_22