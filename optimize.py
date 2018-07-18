import torch
import scipy.optimize
import numpy as np
from torch.autograd import Variable
import logging
logger = logging.getLogger(__name__)

import utils as ut

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
        ut.set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        logger.info('a/e/r: %0.5f, %0.5f, %0.5f' % (actual_improve.item(), expected_improve.item(), ratio.item()))

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            logger.info('fval after: %0.5f'%newfval.item())
            return True, xnew
    return False, x


def function_eval_grad(loss_function,model,*args,params=None):
    # check if need to set model params
    if params is not None:
        ut.set_flat_params_to(model, torch.from_numpy(params))
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

    # Compute loss and extract gradient, loss value
    loss = loss_function(model,*args)
    loss.backward()
    loss_val = loss.detach().double().numpy()
    loss_grad = ut.get_flat_grad_from(model).detach().double().numpy()
    return loss_val,loss_grad


def l2_loss_l2_reg(model,x,y,lambda_penalty):
    """
    Computes ||f(x;theta) - y||^2_2 + lambda sum_i=1^d ||theta_i||_2^2
    """
    x, y = x.detach(), y.detach()
    y_hat = model(x)
    loss = (y_hat - y).pow(2).mean()

    # weight decay
    for param in model.parameters():
        loss += lambda_penalty * param.pow(2).sum()

    return loss


# call l_bfgs(model,loss,*args)
def l_bfgs(fn,model,*args,maxiter=25):
    """
    Run L-BFGS algorithm. args is anything required for the loss function
    """
    _loss_eval_grad = lambda pv : function_eval_grad(fn,model,*args,params=pv)

    params_0 = ut.get_flat_params_from(model).double().numpy()
    params_T, _, opt_info = scipy.optimize.fmin_l_bfgs_b(_loss_eval_grad, params_0, maxiter=maxiter)
    ut.set_flat_params_to(model, torch.from_numpy(params_T))
