import unittest
import numpy as np

from pbcpy.functionals import FunctionalClass
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald

class TestFunctional(unittest.TestCase):
    
    def test_energy(self):
        print()
        print("*"*50)
        print("Testing all energy")
        path_pp='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/OEPP/'
        path_rho='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/rho/'
        file2='As_lda.oe04.recpot'
        file1='Ga_lda.oe04.recpot'
        rhofile='GaAs_rho_test_1.pp'
        mol = PP(filepp=path_rho+rhofile).read()
        optional_kwargs = {}
        optional_kwargs["PP_list"] = [path_pp+file1,path_pp+file1,path_pp+file1,path_pp+file1,path_pp+file1,path_pp+file2,path_pp+file2,path_pp+file2,path_pp+file2,path_pp+file2]
        optional_kwargs["ions"]    = mol.ions 
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)

        func  = IONS.ComputeEnergyDensityPotential(rho=mol.field)
        energy_ie = func.energydensity.integral()

        Ewald_ = ewald(rho = mol.field, ions = mol.ions, verbose = True)
        energy_ii = Ewald_.energy

        thefuncclass = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)
        func  = thefuncclass.ComputeEnergyDensityPotential(rho=mol.field)
        energy_xc = func.energydensity.integral()

        # thefuncclass = FunctionalClass(type='KEDF',name='x_TF_y_vW',is_nonlocal=True)
        thefuncclass = FunctionalClass(type='KEDF',name='WT',is_nonlocal=True)
        func  = thefuncclass.ComputeEnergyDensityPotential(rho=mol.field)
        energy_ke = func.energydensity.integral()

        thefuncclass = FunctionalClass(type='HARTREE',name='HARTREE',is_nonlocal=True)
        func  = thefuncclass.ComputeEnergyDensityPotential(rho=mol.field)
        energy_h = func.energydensity.integral()

        outfile = 'tests/Benchmarks_TOTAL_ENERGY/GaAs_test/output/GaAs_test_1.out'
        E_ke = 0.0
        with open(outfile, 'r') as fr:
            for line in fr :
                if 'ionele  Energy' in line :
                    E_ie = float(line.split()[-1])
                elif 'Kin. Energy' in line :
                    E_ke += float(line.split()[-1])
                elif 'Coulomb Energy' in line :
                    E_h = float(line.split()[-1])
                elif 'Ewald  Energy' in line :
                    E_ii = float(line.split()[-1])
                elif 'Exchange-cor Energy' in line :
                    E_xc = float(line.split()[-1])
                elif 'Total Energy' in line :
                    E_total = float(line.split()[-1])

        Hartree2eV = 27.21138602

        self.assertTrue(np.isclose(energy_ie * Hartree2eV, E_ie, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_ke * Hartree2eV, E_ke, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_h * Hartree2eV, E_h, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_ii * Hartree2eV, E_ii, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_xc * Hartree2eV, E_xc, rtol = 1.E-4))

