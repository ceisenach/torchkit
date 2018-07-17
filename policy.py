# Imports
import torch
from torch.autograd import Variable
import math


class BasePolicy(object):
    """
    Base policy class, supports loading and unloading actor network
    """
    def __init__(self,acnet,**kwargs):
        self._net = acnet

    def save_model(self,path):
        sga_sd = self._net.state_dict()
        torch.save(sga_sd,path)

    def load_model(self,path):
        sd = torch.load(path)
        self._net.load_state_dict(sd)

    def parameters(self):
        return self._net.parameters()

    @property
    def net(self):
        return self._net


class GaussianPolicy(BasePolicy):
    """
    Gaussian Policy.

    Assumes that the network models the mean and the log standard deviation
    of a Gaussian model. The covariance is a diagonal matrix.
    """
    def __init__(self,acnet,**kwargs):
        super(GaussianPolicy,self).__init__(acnet,**kwargs)

    # def nll(self,a_t_hat,s_t=None,a_t = None):
    #     assert(s_t is not None or a_t is not None)
    #     if a_t is None:
    #         a_t = self._net(s_t)
    #     a_t_hat = a_t_hat.unsqueeze(0) if len(a_t_hat.size()) == 1 else a_t_hat
    #     a_t = a_t.unsqueeze(0) if len(a_t.size()) == 1 else a_t

    #     d = a_t_hat.size()[1]
    #     log_prob = - d * math.log(self._sigma) - 0.5*torch.sum((a_t_hat - a_t) ** 2, dim=1) / (self._sigma ** 2)

    #     assert log_prob.dim() == 1
    #     return -log_prob

    def kl_divergence(self,states):
        # states = states.detach()
        mean1, log_std1 = self._net(Variable(states))
        std1 = torch.exp(log_std1)

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def nll(self,a_t_hat,s_t):
        action_means, action_log_stds = self._net(s_t)
        action_stds = torch.exp(action_log_stds)

        var = action_stds.pow(2)
        log_prob = -(a_t_hat - action_means).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - action_log_stds
        log_prob = - log_prob.sum(1, keepdim=True)

        return log_prob


    def action(self,s_t,sample=True):
        # if sample:
        #     self._net.eval()

        action_mean,action_log_std = self._net(s_t)
        action_std = torch.exp(action_log_std)
        if sample:
            action = torch.normal(action_mean, action_std)
            return action.detach()
        return action_mean.detach()


    def fisher_information(self,states,batch_approx=False):
        if batch_approx is True:
            raise RuntimeError('Not implemented yet')

        # Step 1 -- get Jacobian


        # Step 2 -- pre-compute products


        # Step 3 -- return value


    def fisher_vector_product(self,fisher_info):
        pass