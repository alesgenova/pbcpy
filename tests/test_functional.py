import unittest
import numpy as np

from pbcpy.functionals import FunctionalClass
from pbcpy.constants import LEN_CONV
from pbcpy.semilocal_xc import XC,PBE
from pbcpy.formats.qepp import PP

class TestFunctional(unittest.TestCase):
    
    def test_functional(self):
        print()
        print("*"*50)
        print("Testing FunctionalClass XC")
        mol = PP(filepp="tests/Al_fde_rho.pp").read()
        rho_r = mol.field
        thefuncclass = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)
        func2 = thefuncclass.ComputeEnergyPotential(rho=rho_r)
        func1  = XC(density=rho_r,x_str='lda_x',c_str='lda_c_pz',polarization='unpolarized') 
        a = func2.energy
        b = func1.energy
        self.assertTrue(np.isclose(a,b).all())
        # print("Need LibXC Installed - then can run this test")
        # pass

    def test_pbe(self):
        print()
        print("*"*50)
        print("Testing FunctionalClass XC")
        mol = PP(filepp="tests/Al_fde_rho.pp").read()
        rho_r = mol.field
        Functional_LibXC = XC(density=rho_r,x_str='gga_x_pbe',c_str='gga_c_pbe',polarization='unpolarized')
        Functional_LibXC2 = PBE(rho_r,'unpolarized')
        self.assertTrue(np.isclose(Functional_LibXC2.energy,Functional_LibXC.energy))
        # print("Need LibXC Installed - then can run this test")
        # pass
        



