# Imports
import torch
from torch.autograd import Variable
import math

import autodiff as ad
import utils as ut


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

        log_prob = -(a_t_hat - action_means).pow(2) / (2 * var) - action_log_stds #- 0.5 * math.log(2 * math.pi)
        log_prob = - log_prob.sum(1)
        assert log_prob.dim() == 1
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


    def fi2(self,states):
        N = states.shape[0]
        ad.util.zero_jacobian(self._net.parameters())
        act_mean, act_log_std = self._net(states,save_for_jacobian=True)
        act_mean.jacobian(mode='batch',backend='numpy')
        # Dmu = ad.util.gather_jacobian(self._net.parameters())
        # act_mean, act_log_std = self._net(states)
        Dmu = torch.randn(N,self._net._num_outputs,ut.total_params(self._net)-self._net._num_outputs)
        return act_mean,act_log_std,Dmu

    def fisher_information(self,states,batch_approx=False,jacobian_precompute = None):
        # with torch.autograd.no_grad():
        assert states.dim() == 2, "States should be 2D"
        if batch_approx is True:
            raise RuntimeError('Not implemented yet')

        N = states.shape[0]
        # import pdb; pdb.set_trace()

        # Step 1 -- get Jacobian
        act_mean,act_log_std,Dmu = None,None,None
        if jacobian_precompute is None:
            # ad.util.zero_jacobian(self._net.parameters())
            # act_mean, act_log_std = self._net(states,save_for_jacobian=True)
            # act_mean.jacobian(mode='batch')
            # Dmu = ad.util.gather_jacobian(self._net.parameters())
            act_mean, act_log_std = self._net(states)
            Dmu = torch.randn(N,self._net._num_outputs,ut.total_params(self._net)-self._net._num_outputs)
        else:
            act_mean,act_log_std,Dmu = jacobian_precompute['act_mean'], \
            jacobian_precompute['act_log_std'],jacobian_precompute['Dmu']

        # Step 2 -- pre-compute products
        act_mean = act_mean.data
        act_std = torch.exp(act_log_std.data[0])
        act_std_inv = 1./act_std
        act_std_inv2 = act_std_inv ** 2

        Ig_11_Dmu = act_std_inv.unsqueeze(1).unsqueeze(0).expand(Dmu.shape) * Dmu #OK 
        EI_22d = 0.5 * act_std_inv2 #OK
        DmuT = Dmu.transpose_(1,2).contiguous() #OK

        # Step 3 -- return value
        # Not sure if detach is necessary, but want to free any graph
        # return {'DmuT':DmuT.detach(),'Ig_11_Dmu':Ig_11_Dmu.detach(),'EI_22d':EI_22d.detach()}
        # return {'DmuT':DmuT,'Ig_11_Dmu':Ig_11_Dmu,'EI_22d':EI_22d}
        return DmuT.detach(),Ig_11_Dmu.detach(),EI_22d.detach()


    # def fisher_vector_product(self,fisher_info,v):
    #     DmuT,Ig_11_Dmu,EI_22d = fisher_info['DmuT'],fisher_info['Ig_11_Dmu'],fisher_info['EI_22d']
    def fisher_vector_product(self,DmuT,Ig_11_Dmu,EI_22d,v):
        with torch.autograd.no_grad():
            N,d_1,d_2 = DmuT.shape
            v1 = v[:d_1]
            v2 = v[d_1:]

            # calculate upper partition
            v1batched = v1.unsqueeze(0).expand(N,d_1)
            g1 = torch.bmm(Ig_11_Dmu,v1batched.unsqueeze(-1))
            g1 = torch.bmm(DmuT,g1)
            g1 = (1./N) * torch.sum(g1,dim=0).squeeze()

            # calculate lower partition
            g2 = EI_22d * v2

            # combine and return
            g = torch.cat([g1,g2],dim=0)
            return g