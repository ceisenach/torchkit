
import unittest
from . import util as util

class ExtendedTestCase(unittest.TestCase):

    def assertTensorClose(self,g1,g2):
        close,delta_imax,imax,diff_imax = util.tensor_almost_equal(g2,g1,eps=1e-6)
        g1_imax = g1[imax].item()
        g2_imax = g2[imax].item()
        fstr = '\r\nargmax_i delta[i] = %s\r\ndelta[i_max] = %s\r\ndiff[i_max] = %s\r\n'\
                + 'G_1[i_max] = %s\r\nG_2[i_max] = %s\r\nG1: %s\r\nG2: %s\r\n'
        self.assertTrue(close,msg=fstr % (imax,delta_imax,diff_imax,g1_imax,g2_imax,g1,g2))