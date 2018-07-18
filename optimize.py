import torch
import scipy.optimize
import numpy as np
from torch.autograd import Variable
import logging
logger = logging.getLogger(__name__)

from utils import get_flat_grad_from,get_flat_params_from,set_flat_params_to


def conjugate_gradients(Avp, b, nsteps, damping, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p) + p * damping
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def backtracking_ls(model,f,x,fullstep,expected_improve_rate,max_backtracks=10,accept_ratio=.1):
    fval = f(True).data
    logger.info('fval before: %0.5f'%fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        logger.info('a/e/r: %0.5f, %0.5f, %0.5f' % (actual_improve.item(), expected_improve.item(), ratio.item()))

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            logger.info('fval after: %0.5f'%newfval.item())
            return True, xnew
    return False, x


def loss_function_eval_grad(loss_function,args):
    pass


def l2_loss_with_regularizer():
    pass


def _l2_loss_eval_and_grad(value_net,parameter_vals,x,y,l2_weight_penalty):
    set_flat_params_to(value_net, torch.from_numpy(parameter_vals))
    for param in value_net.parameters():
        if param.grad is not None:
            param.grad.data.fill_(0)

    y_hat = value_net(x)

    value_loss = (y_hat - y).pow(2).mean()

    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_weight_penalty
    

    # Compute and extract gradient and loss value
    value_loss.backward()
    loss_val = value_loss.data.double().numpy()
    loss_grad = get_flat_grad_from(value_net).data.double().numpy()

    return loss_val,loss_grad


def l_bfgs(value_net,x,y,l2_penalty):
    # Original code uses the same LBFGS to optimize the value loss
    x = x.detach()
    y = y.detach()
    eval_loss_and_grad = lambda pv : _l2_loss_eval_and_grad(value_net,pv,x,y,l2_penalty)

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(eval_loss_and_grad, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.from_numpy(flat_params))
