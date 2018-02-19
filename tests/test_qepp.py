import unittest
import numpy as np

from pbcpy.formats.qepp import PP
from pbcpy.constants import LEN_CONV

class TestQEPP(unittest.TestCase):
    
    def test_qepp_read(self):
        print()
        print("*"*50)
        print("Testing format.qepp.PP read()")
        dimer = PP(filepp="tests/density_ks.pp").read()
        rho_r = dimer.field
        self.assertAlmostEqual(rho_r.integral(),16)
        
        rho_g = rho_r.fft()
        self.assertAlmostEqual(rho_g[0,0,0,0],16)

        rho_r1 = rho_g.ifft()
        self.assertAlmostEqual(rho_r1.integral(),16)
        self.assertTrue(np.isclose(rho_r1, rho_r, rtol=1.e-4).all())


