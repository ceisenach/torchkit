# Imports
import torch
from torch.autograd import Variable
import logging
logger = logging.getLogger(__name__)

from optimize import l_bfgs, conjugate_gradients,backtracking_ls
from utils import get_flat_params_from,set_flat_params_to

def fisher_vector_product(get_kl,v,model):
    kl = get_kl()
    kl = kl.mean()
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = (flat_grad_kl * Variable(v)).sum()
    grads = torch.autograd.grad(kl_v, model.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

    return flat_grad_grad_kl




class TRPO(object):
    """
    TRPO
    """
    def __init__(self,policy,critic,args):
        self._args = args
        self._policy = policy
        self._actor = policy.net
        self._critic = critic
        self._updates = 0

    def _batch_prepare(self,batch):
        """
        compute advantages for each batch
        """
        value_net,gamma,tau = self._critic, self._args['gamma'], self._args['tau']
        states,actions,masks,rewards = batch
        rewards,masks = rewards.view(-1),masks.view(-1)
        values = value_net(Variable(states))

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
        merge independent batches together
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



    def _trpo_step(self, get_loss, get_kl, max_kl, damping):
        model = self._actor
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        Fvp = lambda v : fisher_vector_product(get_kl,v,model)

        stepdir = conjugate_gradients(Fvp, -loss_grad, 10, damping)

        # originally:      shs = 0.5 * (stepdir * (Fvp(stepdir)+damping*stepdir)).sum(0, keepdim=True)
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        logger.info('lagrange multiplier %s, grad norm: %s' % (str(lm[0]),str(loss_grad.norm())))

        prev_params = get_flat_params_from(model)
        success, new_params = backtracking_ls(model, get_loss, prev_params, fullstep, neggdotstepdir / lm[0])
        set_flat_params_to(model, new_params)

        return loss


    def update(self,batch_list):
        """
        Update the actor and critic net from sampled minibatches
        """
        batch = self._batch_merge(batch_list)
        if batch is None:
            return

        S_t,A_t,G_t,U_t = batch

        # update value net
        l_bfgs(self._critic,S_t,G_t,self._args['l2_pen'])

        fixed_log_prob = -self._policy.nll(Variable(A_t),Variable(S_t)).data.clone()

        # get_loss, get_kl needed to reconstruct graph for higher order gradients
        def get_loss(volatile=False):
            S_t_ = S_t.detach()
            S_t_.requires_grad_(not volatile)
            log_prob = -self._policy.nll(Variable(A_t),S_t_)
            action_loss = -Variable(U_t) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        get_kl = lambda : self._policy.kl_divergence(S_t)

        self._trpo_step(get_loss, get_kl, self._args['max_kl'], self._args['damping'])

