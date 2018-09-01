# Imports
import torch
from torch.autograd import Variable
from torch.distributions.beta import Beta as TBeta
import math
import numpy as np

import autodiff as ad
import utils as ut
from . import BasePolicy


__all__ = ['Gamma']


class Gamma(BasePolicy):
    """
    Gamma Policy.

    
    """
    def __init__(self,acnet,**kwargs):
        super(Gamma,self).__init__(acnet,**kwargs)

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

    def nll(self,a_t_hat,s_t):
        alpha, beta = self._net(s_t.detach())
        lx = torch.log(a_t_hat)
        l1mx = torch.log(1.-a_t_hat)
        lgamma = torch.lgamma(alpha+beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        log_prob = (alpha-1.0)*lx + (beta-1.)*l1mx + lgamma
        nlog_prob = -log_prob.sum(dim=1)
        assert nlog_prob.dim() == 1
        return nlog_prob


    def sample(self,s_t,deterministic=False,**kwargs):
        if deterministic:
            raise RuntimeError('Not supported')
        alpha,beta = self._net(s_t)
        bs = TBeta(alpha,beta)
        return bs.sample().detach()


    def fisher_information(self,states,batch_approx=False,backend='pytorch'):
        # with torch.autograd.no_grad():
        assert states.dim() == 2, "States should be 2D"
        if batch_approx is True:
            raise RuntimeError('Not implemented yet')

        with torch.no_grad():
            fi = None
            if backend == 'pytorch':
                fi = self._fisher_information_torch(states,batch_approx)
            else:
                fi = self._fisher_information_numpy(states,batch_approx)

            return fi

    def _fisher_information_torch(self,states,batch_approx):
        raise NotImplementedError()

    def _fisher_information_numpy(self,states,batch_approx):
        N = states.shape[0]

        # Step 1 -- get Jacobian
        alpha,beta,Dmu = None,None,None
        ad.util.zero_jacobian(self._net.parameters(),backend='numpy')
        alpha, beta = self._net(states,save_for_jacobian=True)
        alpha.jacobian(mode='batch',backend='numpy')
        beta.jacobian(mode='batch',backend='numpy')
        Df = ad.util.gather_jacobian(self._net.alpha.parameters(),backend='numpy')
        Dg = ad.util.gather_jacobian(self._net.beta.parameters(),backend='numpy')
        
        pgab = torch.polygamma(1,alpha.data+beta.data).numpy()
        pga = torch.polygamma(1,alpha.data).numpy()
        pgb = torch.polygamma(1,beta.data).numpy()

        I_11 = np.tile(np.expand_dims(pga-pgab,axis=2),(1,1,Df.shape[2]))
        I_12 = np.tile(np.expand_dims(-pgab,axis=2),(1,1,Df.shape[2]))
        I_22 = np.tile(np.expand_dims(pgb-pgab,axis=2),(1,1,Dg.shape[2]))

        I_11_Df = I_11 * Df
        I_12_Df = I_12 * Df
        I_22_Dg = I_22 * Dg 

        return Df,Dg,I_11_Df,I_12_Df,I_22_Dg


    def fisher_vector_product(self,Df,Dg,I_11_Df,I_12_Df,I_22_Dg,v,backend='pytorch'):
        if backend == 'pytorch':
            raise NotImplementedError()
        else:
            # import pdb; pdb.set_trace()
            Df_T = np.transpose(Df,(0,2,1))
            Dg_T = np.transpose(Dg,(0,2,1))
            I_12_Df_T = np.transpose(I_12_Df,(0,2,1))

            N,d_2,d_1 = Df.shape
            v1 = v[:d_1]
            v2 = v[d_1:]

            # import pdb; pdb.set_trace()
            # expand vectors
            v1batched =  np.expand_dims(v1,axis=0)
            v1batched = np.tile(v1batched,(N,1))
            v1batched = np.expand_dims(v1batched,axis=-1)
            v2batched =  np.expand_dims(v2,axis=0)
            v2batched = np.tile(v2batched,(N,1))
            v2batched = np.expand_dims(v2batched,axis=-1)

            # calculate upper partition
            g1_1 = np.matmul(I_11_Df,v1batched)
            g1_1 = np.matmul(Df_T,g1_1)
            g1_2 = np.matmul(Dg,v2batched)
            g1_2 = np.matmul(I_12_Df_T,g1_2)
            g1 = g1_1 + g1_2
            g1 = (1./N) * np.sum(g1,axis=0).squeeze()

            # calculate lower partition
            g2_1 = np.matmul(I_12_Df,v1batched)
            g2_1 = np.matmul(Dg_T,g2_1)
            g2_2 = np.matmul(I_22_Dg,v2batched)
            g2_2 = np.matmul(Dg_T,g2_2)
            g2 = g2_1 + g2_2
            g2 = (1./N) * np.sum(g2,axis=0).squeeze()

            # combine and return
            g = np.concatenate([g1,g2],axis=0)

            return g

    def scale(self,action,high,low):
        return action