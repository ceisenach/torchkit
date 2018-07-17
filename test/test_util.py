import torch
import numpy as np

from autodiff import util
from . import ExtendedTestCase

class TestKronecker(ExtendedTestCase):

    def test_bkron1(self):
       self._test_bkron_impl(7,3,2,5)

    def test_bkron2(self):
        self._test_bkron_impl(1,3,5,7)

    def test_bkron3(self):
        self._test_bkron_impl(4,3,5,1)

    def _test_bkron_impl(self,n,m,p,q):
        A = torch.randn(n,m)
        B = torch.randn(p,q)

        np_kron = np.kron(A.numpy(),B.numpy())
        tnp_kron = torch.from_numpy(np_kron).unsqueeze(0)
        t_kron = util.bkron(A.unsqueeze(0),B.unsqueeze(0))
        self.assertEqual(torch.sum(torch.abs(tnp_kron-t_kron)),0.0)