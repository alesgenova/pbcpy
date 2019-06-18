import unittest
import numpy as np

from pbcpy.functionals import FunctionalClass
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP

class TestFunctional(unittest.TestCase):
    
    def test_pseudo(self):
        print()
        print("*"*50)
        print("Testing loading pseudopotentials")
        path_pp='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/OEPP/'
        path_rho='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/rho/'
        path_ion='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/pot_ion/'
        file2='As_lda.oe04.recpot'
        file1='Ga_lda.oe04.recpot'
        rhofile='GaAs_rho_test_1.pp'
        ionfile='GaAs_ion_test_1.pp'
        mol = PP(filepp=path_rho+rhofile).read()
        optional_kwargs = {}
        optional_kwargs["PP_list"] = [path_pp+file1,path_pp+file1,path_pp+file1,path_pp+file1,path_pp+file1,path_pp+file2,path_pp+file2,path_pp+file2,path_pp+file2,path_pp+file2]
        optional_kwargs["ions"]    = mol.ions 
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)
        ion_pp = PP(filepp=path_ion+ionfile).read()
        func  = IONS.ComputeEnergyDensityPotential(rho=mol.field)
        a = func.potential
        b = ion_pp.field 
        # self.assertTrue(np.isclose(a,b)).all()
        self.assertTrue(np.allclose(a,b, atol = 1.E-2))


