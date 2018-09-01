# Imports
import torch
from torch.utils import data
import logging
logger = logging.getLogger(__name__)

from . import AlgorithmBase
import optimize as opt
import utils as ut


__all__ = ['TRPO','TRPO_v2']

def fisher_vector_product(get_kl,v,model):
    kl = get_kl()
    kl = kl.mean()
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    v = v.detach()
    v.requires_grad_(True)
    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, model.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

    return flat_grad_grad_kl

def explained_variance(ypred,y):
    """
    FROM: Open AI Baselines
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.dim() == 1 and ypred.dim() == 1
    vary = torch.std(y).item()
    return np.nan if vary==0 else 1 - (torch.std(y-ypred).item()/vary)**2

class TRPOBase(AlgorithmBase):

    def update(self,batch_list):
        """
        Update the actor and critic net from sampled minibatches
        """
        batch = self._batch_merge(batch_list)
        if batch is None:
            return

        S_t,A_t,G_t,U_t_un = batch
        U_t = (U_t_un - U_t_un.mean()) / U_t_un.std()
        logger.debug('Explained Variance VF Before: %0.5f' % explained_variance(self._critic(S_t).view(-1),G_t.view(-1)))

        # update value net
        self._critic_update(S_t,G_t)

        # get_loss, get_kl needed to reconstruct graph for higher order gradients
        with torch.no_grad():
            fixed_log_prob = self._policy.log_likelihood(A_t,S_t)
            output_old = self._actor(S_t)

        def get_loss(volatile=False):
            log_prob = self._policy.log_likelihood(A_t,S_t)
            action_loss = - U_t.view(-1) * torch.exp(log_prob - fixed_log_prob)
            return action_loss.mean()

        get_kl = lambda : self._policy.kl_divergence(S_t,*output_old)

        # do actor step
        self._trpo_step(get_loss, get_kl, self._args['max_kl'], self._args['damping'])
        logger.debug('Mean KL: %0.5f' % get_kl().mean().item())




class TRPO(TRPOBase):
    """
    TRPO. Original implementation from TRPO paper. Pytorch version of code found in
    John Shulman's modular_rl repository on github.
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(TRPO,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_gae_td0_return


    def _trpo_step(self, get_loss, get_kl, max_kl, damping):
        model = self._actor
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        Fvp = lambda v : fisher_vector_product(get_kl,v,model)
        stepdir = opt.conjugate_gradients(Fvp, -loss_grad, 10, damping)
        shs = 0.5 * (stepdir * (Fvp(stepdir)+damping*stepdir)).sum(0)
        lm = torch.sqrt(shs / max_kl).item()
        fullstep = stepdir / lm

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        # import pdb; pdb.set_trace()
        logger.debug('lagrange multiplier %s, grad norm: %s' % (str(lm),str(loss_grad.norm())))

        prev_params = ut.get_flat_params_from(model)
        success, new_params = opt.backtracking_ls_ratio(model, get_loss, prev_params, fullstep, neggdotstepdir / lm)
        ut.set_flat_params_to(model, new_params)

        return loss

    def _critic_update(self,S_t,G_t):
        opt.l_bfgs(opt.l2_loss_l2_reg,self._critic,S_t,G_t,self._args['l2_pen'])



class TRPO_v2(TRPOBase):
    """
    TRPO: Pytorch version of TRPO as implemented in TRPO_MPI in OpenAI Baselines.
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(TRPO_v2,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_gae_lambda_return
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._args['lr_critic'])

    def _trpo_step(self, get_loss, get_kl, max_kl, damping):
        model = self._actor
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        Fvp = lambda v : fisher_vector_product(get_kl,v,model)
        stepdir = opt.conjugate_gradients(Fvp, -loss_grad, 10, damping)
        shs = 0.5 * (stepdir * (Fvp(stepdir)+damping*stepdir)).sum(0)

        lm = torch.sqrt(shs / max_kl).item()
        fullstep = stepdir / lm
        expected_improve = (-loss_grad * stepdir).sum(0, keepdim=True)
        logger.debug('lagrange multiplier %5.3g, grad norm: %5.3g' % (lm,loss_grad.norm().item()))

        prev_params = ut.get_flat_params_from(model)
        kl_constraint_eval = lambda : get_kl().mean().item()
        success, new_params = opt.backtracking_ls(model, get_loss, prev_params, fullstep, expected_improve, kl_constraint_eval, 1.5 * max_kl)
        ut.set_flat_params_to(model, new_params)

        return loss

    def _critic_update(self,S_t,G_t,num_epochs=5):
        dataset = data.TensorDataset(S_t.detach(),G_t.detach())
        dl = data.DataLoader(dataset,batch_size=64,shuffle=True)
        for _ in range(num_epochs):
            for s,g in dl:
                v = self._critic(s)
                lf_critic = torch.mean((g - v)**2)
                self._critic_optimizer.zero_grad()
                lf_critic.backward()
                self._critic_optimizer.step()