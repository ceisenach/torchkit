import torch

from . import AlgorithmBase
import autodiff as ad

class NACGauss(AlgorithmBase):
    """
    Natural Actor-Critic for Gaussian Family
    """
    def __init__(self,policy,critic,args,**kwargs):
        super(NACGauss,self).__init__(policy,critic,args,**kwargs)
        self._updates = 0
        self._batch_prepare = self._batch_prepare_full_gae
        self._critic_optimizer = torch.optim.SGD(self._critic.parameters(), lr=args['lr'])


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


    def _actor_update(self,S_t,A_t_hat,U_t):
        # dont need to construct loss, can just use jacobian directly
        act_mean, act_log_std = self._net(S_t,save_for_jacobian=True)
        act_mean.jacobian(mode='batch')
        Dmu = ad.util.gather_jacobian(self._net.parameters())
        



        lf_actor = torch.mean(U_1.view(-1) * self._policy.nll(a_t_hat,a_t = a_t))




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

        # update
        self._updates += 1
