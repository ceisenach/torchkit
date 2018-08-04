import torch

from . import AlgorithmBase

__all__ = ['A2C']

class A2C(AlgorithmBase):
    """
    A2C: Synchronous version of A3C
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(A2C,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_advantages
        self._critic_optimizer = torch.optim.SGD(self._critic.parameters(), lr=args['lr_critic'])
        self._actor_optimizer = torch.optim.SGD(self._actor.parameters(), lr=args['lr_actor'])

    def update(self,batch_list):
        """
        Update the actor and critic net from sampled minibatches
        """
        batch = self._batch_merge(batch_list)
        if batch is None:
            return

        self._policy.net.train()
        S_t,A_t_hat,G_t,U_t = batch
        V_t = self._critic(S_t.detach())

        # critic update
        self._critic_optimizer.zero_grad()
        lf_critic = torch.mean((G_t - V_t)**2)
        lf_critic.backward()
        self._critic_optimizer.step()

        # actor loss
        self._actor_optimizer.zero_grad()
        lf_actor = torch.mean(U_t.view(-1) * self._policy.nll(A_t_hat,S_t.detach()))
        lf_actor.backward()
        self._actor_optimizer.step()

        # update
        self._updates += 1


class A2C_GAE(A2C):
    def __init__(self,policy,critic,args,**kwargs):
        super(A2C_GAE,self).__init__(policy,critic,args,**kwargs)
        self._batch_prepare = self._batch_prepare_gae_lambda_return