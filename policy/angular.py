# Imports
import torch
from torch.autograd import Variable
from . import BasePolicy
import math

__all__ = ['AngularGaussML']

EPS = 10**(-6)


class AngularGaussML(BasePolicy):
    """
    Angular Gaussian Policy -- negative log-likelihood is with respect to the angular Gaussian density.

    Assumes that the network models the mean and the log standard deviation
    of a Gaussian model. The covariance is a diagonal matrix.
    """

    def __init__(self,acnet,sigma = 0.2):
        super(AngularGaussML,self).__init__(acnet)
        self._sigma = sigma

    # gets M_{d-1} and M_{d-2}
    def _m_function(self,d, alpha):
        assert(d > 1)
        if type(alpha) == Variable:
            # detach alpha, handle the differential by ourselves
            alpha = Variable(alpha.data)
        def normal01_pdf(x):
            return torch.exp(-(x ** 2) / 2.0) / math.sqrt(2 * math.pi)
        def normal01_cdf(x):
            return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

        m0 = normal01_cdf(alpha)
        m1 = alpha * m0 + normal01_pdf(alpha)
        m = m1

        for i in range(d-1):
            m = alpha * m1 + (d + 1.0) * m0
            m_grad = (d + 1.0) * m1
            m0, m1 = m1, m

        return m1,m0

    def sample(self,s_t,deterministic=False,**kwargs):
        action_mean,_ = self._net(s_t)
        if not deterministic:
            action = torch.normal(action_mean, self._sigma)
            return action.detach()
        return action_mean.detach()


    def log_likelihood(self,a_t_hat,s_t):
        a_t,log_std_t = self._net(s_t)

        a_t_hat = a_t_hat.unsqueeze(0) if len(a_t_hat.size()) == 1 else a_t_hat
        a_t = a_t.unsqueeze(0) if len(a_t.size()) == 1 else a_t
        log_std_t = log_std_t.unsqueeze(0) if len(log_std_t.size()) == 1 else log_std_t
        std_t = torch.exp(log_std_t)
        var_t = std_t ** 2

        # basic expressions in angular gaussian log prob
        d = a_t_hat.size()[1]
        xTSmu = torch.sum(a_t_hat * a_t / var_t + EPS, dim=1)
        xTSx = torch.sum(a_t_hat ** 2 / var_t, dim=1)
        muTSmu = torch.sum(a_t ** 2 / var_t, dim=1)
        xTSx_sqrt = torch.sqrt(xTSx)
        alpha = (1./xTSx_sqrt) * xTSmu
        alpha_sq = alpha ** 2

        # \cM_{d-1}(\alpha)
        m_dm1, m_dm2 = self._m_function(d, alpha)

        # higher level expressions
        term_3 = 0.5 * (alpha_sq - muTSmu)
        term_4 = (d * m_dm2 / m_dm1) * alpha
        log_prob = term_3 + term_4

        assert log_prob.dim() == 1
        return log_prob
