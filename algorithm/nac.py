import torch


from . import AlgorithmBase


class NACGauss(AlgorithmBase):
    """
    Natural Actor-Critic for Gaussian Family
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(NACGauss,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_full_gae


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
            # S_t_ = S_t.detach()
            # S_t_.requires_grad_(not volatile)
            log_prob = -self._policy.nll(Variable(A_t),S_t_)
            action_loss = -Variable(U_t) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        get_kl = lambda : self._policy.kl_divergence(S_t)

        self._trpo_step(get_loss, get_kl, self._args['max_kl'], self._args['damping'])

