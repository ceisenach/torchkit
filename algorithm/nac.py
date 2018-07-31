import torch
import numpy as np

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
        self._batch_prepare = self._batch_prepare_advantages
        self._critic_optimizer = torch.optim.SGD(self._critic.parameters(), lr=args['lr'])


    def _actor_update(self,S_t,A_t_hat,U_t):
        lg = None
        with torch.enable_grad():
            # Get loss gradient
            lf_actor = torch.mean(U_t.view(-1) * self._policy.nll(A_t_hat,S_t.detach()))
            grads = torch.autograd.grad(lf_actor, self._policy.parameters())
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
            prev_params = ut.get_flat_params_from(self._actor)
            # # weight decay
            # l2_pen = 0.01
            # stepdir = stepdir + l2_pen * prev_params
            new_params = prev_params - self._args['lr'] * stepdir
            ut.set_flat_params_to(self._actor, new_params)



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
