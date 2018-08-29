import torch
import scipy.optimize
import numpy as np
from torch.autograd import Variable
import logging
logger = logging.getLogger(__name__)

import utils as ut

def conjugate_gradients(Avp, b, nsteps, damping, residual_tol=1e-10, grad=True, backend='pytorch'):
    with torch.set_grad_enabled(grad):
        x = torch.zeros(b.shape) if backend == 'pytorch' else np.zeros(b.shape,dtype=np.float32)
        df = torch.dot if backend == 'pytorch' else np.dot
        nf = torch.norm if backend =='pytorch' else np.linalg.norm
        r = b.clone() if backend == 'pytorch' else b.copy()
        p = b.clone() if backend == 'pytorch' else b.copy()

        fmtstr =  "%10i %10.3g %10.3g"
        titlestr =  "%10s %10s %10s"
        logger.debug(titlestr % ("iter", "residual norm", "soln norm"))

        rdotr = df(r, r)
        for i in range(nsteps):
            logger.debug(fmtstr % (i, rdotr, nf(x)))
            _Avp = Avp(p) + p * damping
            alpha = rdotr / df(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = df(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        logger.debug(fmtstr % (i+1, rdotr, nf(x)))
        # logger.info('CG steps ')
        return x



def backtracking_ls_ratio(model,f,x,fullstep,expected_improve_rate,max_backtracks=10,accept_ratio=.1):
    fval = f(True).data
    logger.debug('fval before: %0.5f'%fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        ut.set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        logger.debug('Backtrack iter %d: a/e/r: %0.5f, %0.5f, %0.5f' % (_n_backtracks,actual_improve.item(), expected_improve.item(), ratio.item()))

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            logger.debug('Backtrack iter %d: Done -- fval after: %0.5f' % (_n_backtracks, newfval.item()))
            return True, xnew
    return False, x



def backtracking_ls(model,f,x,fullstep,expected_improve,get_constraint = lambda x : -1,constraint_max = 0,max_backtracks=10):
    fval = f(True).data
    logger.debug('fval before: %0.5f'%fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        ut.set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        cv = get_constraint() > constraint_max
        ic = actual_improve.item() < 0
        logger.debug('Backtrack iter %d: a/e/cv: %0.5f, %0.5f, %0.5f' % (_n_backtracks, actual_improve.item(), expected_improve.item(),cv))
        if cv:
            import pdb; pdb.set_trace()
            logger.debug('Backtrack iter %d: Constraint Violated %0.5f' % (_n_backtracks,get_constraint()))
        elif actual_improve.item() < 0:
            logger.debug('Backtrack iter %d: No Improvement' % _n_backtracks)
        else:
            logger.debug('Backtrack iter %d: Done -- fval after: %0.5f' % (_n_backtracks, newfval.item()))
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
