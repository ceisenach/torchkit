import torch

from . import AlgorithmBase


class A2C(AlgorithmBase):
    """
    Natural Actor-Critic for Gaussian Family
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(A2C,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_advantages
        self._critic_optimizer = torch.optim.SGD(self._critic.parameters(), lr=args['lr'])
        self._actor_optimizer = torch.optim.SGD(self._actor.parameters(), lr=args['lr'])


    def _batch_prepare_advantages(self,batch):
        # Compute advantages
        self._actor.eval()
        S,A,M,R = batch
        s_T = S[-1].detach().unsqueeze(0)
        r_tp1 = R[:-1].view(-1,1)

        U = torch.zeros((r_tp1.size()[0]+1,1))
        qT = self._critic(s_T)
        U[-1,:] = qT.data

        length_episode = U.size()[0]
        for i in range(length_episode-1):
            U[length_episode-i-2,:] = r_tp1[length_episode-i-2,:] + self._args['gamma'] * U[length_episode-i-1,:] * M[length_episode-i-2,:]
        U = U[:-1]
        return S[:-1],A[:-1],U,U


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
        lf_critic = torch.mean((G_t.detach() - V_t)**2)
        lf_critic.backward()
        self._critic_optimizer.step()

        # actor loss
        self._actor_optimizer.zero_grad()
        U = (U_t.detach() - V_t.detach())
        lf_actor = torch.mean(U.view(-1) * self._policy.nll(A_t_hat,S_t.detach()))
        lf_actor.backward()
        self._actor_optimizer.step()

        # update
        self._updates += 1
