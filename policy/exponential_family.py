# Imports
import torch
from torch.autograd import Variable
from torch.distributions.beta import Beta as TBeta
import math
import numpy as np

import autodiff as ad
import utils as ut

from . import BasePolicy

__all__ = ['ExponentialFamily2P']



class ExponentialFamily2P(BasePolicy):
    """
    Two Parameter Exponential Families with independent components.
    Fisher information should be block-diagonal.

    Handles boiler-plate / common operations
    """
    def __init__(self,acnet,**kwargs):
        super(ExponentialFamily2P,self).__init__(acnet,**kwargs)


    def fisher_information_params(self,param_1,param_2,backend='pytorch'):
        """
        Fisher information in terms of parameter values. Returns a batch of diagonals of each sub-matrix.

        param_1 and param_2 are of type torch.Tensor
        """
        raise NotImplementedError()

    def fisher_information(self,states,batch_approx=False,backend='pytorch'):
        """
        Pre-compute quantities used for Fisher Information / Fisher Vector product.

        Returns Df, Dg, I_11 * Df,I_12 * Df,I_22 * Dg

        batch_approx -- if true lower memory approach is used (expectation is moved inside multiplication)
        """
        assert states.dim() == 2, "States should be 2D"
        if batch_approx is True:
            raise RuntimeError('Not implemented yet')

        # Step 1 -- get Jacobian
        ad.util.zero_jacobian(self._net.parameters(),backend=backend)
        import pdb; pdb.set_trace()
        param_1, param_2 = self._net(states,save_for_jacobian=True)
        param_1.jacobian(mode='batch',backend=backend)
        param_2.jacobian(mode='batch',backend=backend)
        Df = ad.util.gather_jacobian(self._net.param_1.parameters(),backend=backend)
        Dg = ad.util.gather_jacobian(self._net.param_2.parameters(),backend=backend)
        
        I_11,I_12,I_22 = self.fisher_information_params(param_1.data,param_2.data,backend=backend)

        if backend == 'pytorch':
            # import pdb; pdb.set_trace()
            I_11 = I_11.unsqueeze(-1).expand(Df.shape)
            I_12 = I_12.unsqueeze(-1).expand(Df.shape)
            I_22 = I_22.unsqueeze(-1).expand(Dg.shape)
        elif  backend == 'numpy':
            I_11 = np.tile(np.expand_dims(I_11,axis=2),(1,1,Df.shape[2]))
            I_12 = np.tile(np.expand_dims(I_12,axis=2),(1,1,Df.shape[2]))
            I_22 = np.tile(np.expand_dims(I_22,axis=2),(1,1,Dg.shape[2]))
        else:
            raise RuntimeError()


        I_11_Df = I_11 * Df
        I_12_Df = I_12 * Df
        I_22_Dg = I_22 * Dg 

        return Df,Dg,I_11_Df,I_12_Df,I_22_Dg



    def fisher_vector_product(self,Df,Dg,I_11_Df,I_12_Df,I_22_Dg,v,backend='pytorch'):
        if backend == 'pytorch':
            # Df_T = Df.transpose(1,2).contiguous()
            # Dg_T = Dg.transpose(1,2).contiguous()
            # I_12_Df = I_12_Df.transpose(1,2).contiguous()
            Df_T = Df.transpose(1,2)
            Dg_T = Dg.transpose(1,2)
            I_12_Df_T = I_12_Df.transpose(1,2)

            v = v.detach()
            N,d_2,d_1 = Df.shape
            _,_,d_1_g = Dg.shape
            v1 = v[:d_1]
            v2 = v[d_1:]
            # import pdb; pdb.set_trace()
            v1batched = v1.unsqueeze(0).expand(N,d_1)
            v2batched = v2.unsqueeze(0).expand(N,d_1_g)

            # calculate upper partition
            g1_1 = torch.bmm(I_11_Df,v1batched.unsqueeze(-1))
            g1_1 = torch.bmm(Df_T,g1_1)
            g1_2 = torch.bmm(Dg,v2batched.unsqueeze(-1))
            g1_2 = torch.bmm(I_12_Df_T,g1_2)
            g1 = g1_1 + g1_2
            g1 = (1./N) * torch.sum(g1,dim=0).squeeze()

            # calculate lower partition
            g2_1 = torch.bmm(I_12_Df,v1batched.unsqueeze(-1))
            g2_1 = torch.bmm(Dg_T,g2_1)
            g2_2 = torch.bmm(I_22_Dg,v2batched.unsqueeze(-1))
            g2_2 = torch.bmm(Dg_T,g2_2)
            g2 = g2_1 + g2_2
            g2 = (1./N) * torch.sum(g2,dim=0).squeeze()

            # combine and return
            g = torch.cat([g1,g2],dim=0).data
            return g

        elif backend == 'numpy':
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
        else:
            raise RuntimeError()
