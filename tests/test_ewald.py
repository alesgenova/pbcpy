import unittest
import numpy as np

from pbcpy.functionals import FunctionalClass
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald
import time

class TestEwald(unittest.TestCase):
    
    def test_ewald(self):
        print()
        print("*"*50)
        print("Testing Ewald energy")
        path_pp='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/OEPP/'
        path_rho='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/rho/'
        path_ion='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/pot_ion/'
        file2='As_lda.oe04.recpot'
        file1='Ga_lda.oe04.recpot'
        rhofile='GaAs_rho_test_1.pp'
        ionfile='GaAs_ion_test_1.pp'
        mol = PP(filepp=path_rho+rhofile).read()
        Ewald_ = ewald(rho = mol.field, ions = mol.ions, verbose = False)
        Ewald_PME = ewald(rho = mol.field, ions = mol.ions, verbose = False, PME = True)

        self.assertTrue(np.allclose(Ewald_.energy, Ewald_PME.energy, atol = 1.E-5))
        self.assertTrue(np.allclose(Ewald_.forces, Ewald_PME.forces, atol = 1.E-5))
        self.assertTrue(np.allclose(Ewald_.stress, Ewald_PME.stress, atol = 1.E-5))


