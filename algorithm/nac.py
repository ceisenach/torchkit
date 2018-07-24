import torch

from . import AlgorithmBase
import autodiff as ad
import optimize as opt
import utils as ut
import gc

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
        # # dont need to construct loss, can just use jacobian directly
        # act_mean, act_log_std = self._actor(S_t,save_for_jacobian=True)
        # # act_mean.jacobian(mode='batch')
        # # Dmu = ad.util.gather_jacobian(self._net.parameters())
        
        # Get loss gradient
        ad.util.zero_grad(self._actor.parameters())
        lf_actor = torch.mean(U_t.view(-1) * self._policy.nll(A_t_hat,S_t))
        lf_actor.backward()
        lg = ad.util.gather_grad(self._actor.parameters())

        # Get natural gradient direction
        # fi = self._policy.fisher_information(S_t)
        # FVP = lambda v : self._policy.fisher_vector_product(fi,v)
        # stepdir = opt.conjugate_gradients(FVP, lg, 10, self._args['damping'])

        fi_a,fi_b,fi_c = self._policy.fisher_information(S_t)
        FVP = lambda v : self._policy.fisher_vector_product(fi_a,fi_b,fi_c,v)
        stepdir = opt.conjugate_gradients(FVP, lg, 10, self._args['damping'])
        stepdir = lg

        # Update model
        prev_params = ut.get_flat_params_from(self._actor)
        new_params = prev_params - self._args['lr'] * stepdir
        ut.set_flat_params_to(self._actor, new_params)
        gc.collect()

        # del fi,lg,FVP,stepdir,prev_params,new_params,lf_actor

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
