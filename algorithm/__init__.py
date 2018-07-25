import torch

#######################################
# Core Classes

class AlgorithmBase(object):

    def __init__(self,policy,critic,args,**kwargs):
        self._args = args
        self._policy = policy
        self._actor = policy.net
        self._critic = critic


    def _batch_prepare(self,batch):
        raise NotImplementedError()


    def _batch_prepare_full_gae(self,batch):
        """
        compute advantages for each batch using GAE on full trajectories -- could be high variance
        """
        value_net,gamma,tau = self._critic, self._args['gamma'], self._args['tau']
        states,actions,masks,rewards = batch
        rewards,masks = rewards.view(-1),masks.view(-1)
        values = value_net(states)

        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]


        advantages = (advantages - advantages.mean()) / advantages.std()

        return states,actions,returns,advantages


    def _batch_merge(self,batch_list):
        """
        merge independent batches together -- should always return a 4-tuple
        """
        if not isinstance(batch_list[0],list):
            return self._batch_prepare(batch_list)
        S,A,R,U = [],[],[],[]
        for b in filter(lambda bt: bt[0] is not None, batch_list):
            s,a,r,u = self._batch_prepare(b)
            S.append(s)
            A.append(a)
            U.append(u)
            R.append(r)
        if len(S) == 0:
            return None
        return torch.cat(S,dim=0),torch.cat(A,dim=0),torch.cat(R,dim=0),torch.cat(U,dim=0)




###########################################
# Import commonly used names
from .trpo import TRPO
from .nac import NACGauss
from .a2c import A2C