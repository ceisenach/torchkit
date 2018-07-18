# Imports
import torch
import logging
logger = logging.getLogger(__name__)

from . import AlgorithmBase
import optimize as opt
import utils as ut

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


class TRPO(AlgorithmBase):
    """
    TRPO
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(TRPO,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_full_gae


    def _trpo_step(self, get_loss, get_kl, max_kl, damping):
        model = self._actor
        loss = get_loss()
        grads = torch.autograd.grad(loss, model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        Fvp = lambda v : fisher_vector_product(get_kl,v,model)

        stepdir = opt.conjugate_gradients(Fvp, -loss_grad, 10, damping)

        # originally:      shs = 0.5 * (stepdir * (Fvp(stepdir)+damping*stepdir)).sum(0, keepdim=True)
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        logger.info('lagrange multiplier %s, grad norm: %s' % (str(lm[0]),str(loss_grad.norm())))

        prev_params = ut.get_flat_params_from(model)
        success, new_params = opt.backtracking_ls(model, get_loss, prev_params, fullstep, neggdotstepdir / lm[0])
        ut.set_flat_params_to(model, new_params)

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
        opt.l_bfgs(opt.l2_loss_l2_reg,self._critic,S_t,G_t,self._args['l2_pen'])

        fixed_log_prob = -self._policy.nll(A_t,S_t).data.clone()

        # get_loss, get_kl needed to reconstruct graph for higher order gradients
        def get_loss(volatile=False):
            log_prob = -self._policy.nll(A_t,S_t)
            action_loss = - U_t * torch.exp(log_prob - fixed_log_prob)
            return action_loss.mean()

        get_kl = lambda : self._policy.kl_divergence(S_t)

        self._trpo_step(get_loss, get_kl, self._args['max_kl'], self._args['damping'])