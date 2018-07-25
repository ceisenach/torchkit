import torch
import numpy as np

from . import AlgorithmBase
import autodiff as ad
import optimize as opt
import utils as ut
import gc



# def fisher_vector_product(self,fisher_info,v):
#     DmuT,Ig_11_Dmu,EI_22d = fisher_info['DmuT'],fisher_info['Ig_11_Dmu'],fisher_info['EI_22d']
def fisher_vector_product(DmuT,Ig_11_Dmu,EI_22d,v,backend='pytorch'):
    if backend == 'pytorch':
        v = v.detach()
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
        g = torch.cat([g1,g2],dim=0).data

        return g
    else:
        N,d_1,d_2 = DmuT.shape
        v1 = v[:d_1]
        v2 = v[d_1:]

        # calculate upper partition
        v1batched =  np.expand_dims(v1,axis=0)
        v1batched = np.tile(v1batched,(N,1))
        v1batched = np.expand_dims(v1batched,axis=-1)
        g1 = np.matmul(Ig_11_Dmu,v1batched)
        g1 = np.matmul(DmuT,g1)
        g1 = (1./N) * np.sum(g1,axis=0).squeeze()

        # calculate lower partition
        g2 = EI_22d * v2

        # combine and return
        g = np.concatenate([g1,g2],axis=0)

        return g


def fisher_vec_id(a,b,c,v):
    d = a*2
    return v*2


class NACGauss(AlgorithmBase):
    """
    Natural Actor-Critic for Gaussian Family
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(NACGauss,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_full_gae
        self._critic_optimizer = torch.optim.SGD(self._critic.parameters(), lr=args['lr'])


    def _actor_update(self,S_t,A_t_hat,U_t):
        lg = None
        with torch.enable_grad():
            # Get loss gradient
            lf_actor = torch.mean(U_t.view(-1) * self._policy.nll(A_t_hat,S_t))
            grads = torch.autograd.grad(lf_actor, self._policy.parameters())
            lg = torch.cat([grad.view(-1) for grad in grads]).data
            lg = lg.numpy()

        with torch.no_grad():
            # Get natural gradient direction
            fi = self._policy.fisher_information(S_t,backend='numpy')
            # fi = [torch.Tensor(a).clone() for a in fi]
            FVP = lambda v : fisher_vector_product(*fi,v,backend='numpy')
            # FVP = lambda v : fisher_vec_id(*fi,v)
            b = lg
            for i in range(10):
                b = FVP(lg)
            # stepdir = opt.conjugate_gradients(FVP, lg, 10, self._args['damping'],grad=False)
            # del FVP
            # for i in fi:
            #     del i
            # Update model
            # import pdb; pdb.set_trace()
            stepdir = lg
            prev_params = ut.get_flat_params_from(self._actor)
            new_params = prev_params - self._args['lr'] * torch.Tensor(stepdir)
            ut.set_flat_params_to(self._actor, new_params)


    def _actor_update_simple(self,S_t,A_t_hat,U_t):
        # # dont need to construct loss, can just use jacobian directly
        # act_mean, act_log_std = self._actor(S_t,save_for_jacobian=True)
        # # act_mean.jacobian(mode='batch')
        # # Dmu = ad.util.gather_jacobian(self._net.parameters())
        
        # Get loss gradient
        ad.util.zero_grad(self._actor.parameters())
        lf_actor = torch.mean(U_t.view(-1) * self._policy.nll(A_t_hat,S_t))
        lf_actor.backward()
        lg = ad.util.gather_grad(self._actor.parameters())

        prev_params = ut.get_flat_params_from(self._actor)
        new_params = prev_params - self._args['lr'] * lg
        ut.set_flat_params_to(self._actor, new_params)

        # del fi,lg,FVP,stepdir,prev_params,new_params,lf_actor



    def update(self,batch_list):
        """
        Update the actor and critic net from sampled minibatches
        """
        # self._actor.train(),self._critic.train()
        batch = self._batch_merge(batch_list)
        if batch is None:
            return

        S_t,A_t_hat,G_t,U_t = batch
        V_t = self._critic(S_t)

        # critic update
        lf_critic = torch.mean((G_t - V_t)**2)
        self._critic_optimizer.zero_grad()
        lf_critic.backward()
        self._critic_optimizer.step()

        # actor loss
        self._actor_update(S_t,A_t_hat,U_t)
        # self._actor_update_simple(S_t,A_t_hat,U_t)

        # update
        self._updates += 1
