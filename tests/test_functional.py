import unittest
import numpy as np

from pbcpy.functionals import FunctionalClass
from pbcpy.constants import LEN_CONV
from pbcpy.functionals import FunctionalClass
from pbcpy.semilocal_xc import XC,PBE

class TestFunctional(unittest.TestCase):
    
    def test_functional(self):
        print()
        print("*"*50)
        print("Testing FunctionalClass XC")
        dimer = PP(filepp="tests/density_ks.pp").read()
        rho_r = dimer.field
        thefuncclass = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)
        func2 = thefuncclass.ComputeEnergyDensityPotential(rho=rho_r)
        func1  = XC(density=rho_r,x_str='lda_x',c_str='lda_c_pz',polarization='unpolarized') 
        a = func2.energydensity
        b = func1.energydensity
        self.assertTrue(np.isclose(a,b).all())

    def test_pbe(self):
        print()
        print("*"*50)
        print("Testing FunctionalClass XC")
        dimer = PP(filepp="tests/density_ks.pp").read()
        rho_r = dimer.field
        Functional_LibXC = XC(density=rho_r,x_str='gga_x_pbe',c_str='gga_c_pbe',polarization='unpolarized')
        Functional_LibXC2 = PBE(rho_r,'unpolarized')
        a = np.sum( - )
        self.assertTrue(np.isclose(Functional_LibXC2.energydensity,Functional_LibXC.energydensity).all())




