import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)

from . import AlgorithmBase
import autodiff as ad
import optimize as opt
import utils as ut
import gc

__all__ = ['NAC','NAC_LS']

class NAC(AlgorithmBase):
    """
    Natural Actor-Critic for Gaussian Family
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(NAC,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_gae_lambda_return
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args['lr_critic'])

    def _actor_update(self,S_t,A_t_hat,U_t):
        lg = None
        with torch.enable_grad():
            # Get loss gradient
            lf_actor = - torch.mean(U_t.view(-1) * self._policy.log_likelihood(A_t_hat,S_t.detach()))
            # import pdb; pdb.set_trace()
            grads = torch.autograd.grad(lf_actor, self._policy.parameters(ordered=True))
            lg = torch.cat([grad.view(-1) for grad in grads]).data
            lg = lg if self._args['backend'] == 'pytorch' else lg.numpy()

        # Get natural gradient direction
        # Disable grad to make sure no graph is made
        with torch.no_grad():
            fi = self._policy.fisher_information(S_t,backend=self._args['backend'])
            FVP = lambda v : self._policy.fisher_vector_product(*fi,v,backend=self._args['backend'])
            stepdir = opt.conjugate_gradients(FVP, lg, 10, self._args['damping'],grad=False,backend=self._args['backend'])
            if np.isnan(stepdir).any() and self._args['debug']:
                import pdb; pdb.set_trace()
            stepdir = stepdir if isinstance(stepdir,torch.Tensor) else torch.from_numpy(stepdir)
            prev_params = ut.get_flat_params_from(self._actor,ordered=True)
            # # weight decay
            # l2_pen = 0.01
            # stepdir = stepdir + l2_pen * prev_params
            new_params = prev_params - self._args['lr_actor'] * stepdir
            ut.set_flat_params_to(self._actor, new_params,ordered=True)



    def update(self,batch_list):
        """
        Update the actor and critic net from sampled minibatches
        """
        batch = self._batch_merge(batch_list)
        if batch is None:
            return

        S_t,A_t_hat,G_t,U_t = batch
        V_t = self._critic(S_t.detach())

        # critic update
        lf_critic = torch.mean((G_t - V_t)**2)
        self._critic_optimizer.zero_grad()
        lf_critic.backward()
        self._critic_optimizer.step()

        # actor loss
        self._actor_update(S_t,A_t_hat,U_t)

        # update
        self._updates += 1



class NAC_LS(NAC):

    def _actor_update(self,S_t,A_t_hat,U_t):
        lg = None
        with torch.enable_grad():
            # Get loss gradient
            lf_actor = - torch.mean(U_t.view(-1) * self._policy.log_likelihood(A_t_hat,S_t.detach()))
            grads = torch.autograd.grad(lf_actor, self._policy.parameters(ordered=True))
            lg_t = torch.cat([grad.view(-1) for grad in grads]).data
            lg = lg_t if self._args['backend'] == 'pytorch' else lg_t.numpy()

        # Get natural gradient direction
        # Disable grad to make sure no graph is made
        with torch.no_grad():
            fi = self._policy.fisher_information(S_t,backend=self._args['backend'])
            FVP = lambda v : self._policy.fisher_vector_product(*fi,v,backend=self._args['backend'])
            stepdir = opt.conjugate_gradients(FVP, lg, 10, self._args['damping'],grad=False,backend=self._args['backend'])
            if np.isnan(stepdir).any() and self._args['debug']:
                import pdb; pdb.set_trace()
            stepdir = stepdir if isinstance(stepdir,torch.Tensor) else torch.from_numpy(stepdir)
            prev_params = ut.get_flat_params_from(self._actor,ordered=True)

            # Compute Max Step-size
            natural_norm = torch.sqrt((stepdir * (FVP(stepdir)+self._args['damping']*stepdir)).sum(0)).item()
            max_step_size = self._args['lr_actor'] / natural_norm

            # # weight decay
            # l2_pen = 0.01
            # stepdir = stepdir + l2_pen * prev_params
            
            logger.debug('Max Step Size %5.3g, Loss Grad Norm: %5.3g, Natural Norm: %5.3g' % (max_step_size,lg_t.norm().item(),natural_norm))
            new_params = prev_params - max_step_size*stepdir
            ut.set_flat_params_to(self._actor, new_params,ordered=True)
