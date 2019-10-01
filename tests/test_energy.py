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
        print("Testing loading pseudopotentials")
        path_pp='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/OEPP/'
        path_rho='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/rho/'
        file1='Ga_lda.oe04.recpot'
        file2='As_lda.oe04.recpot'
        rhofile='GaAs_rho_test_1.pp'
        mol = PP(filepp=path_rho+rhofile).read()
        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Ga': path_pp+file1,'As': path_pp+file2}
        optional_kwargs["ions"]    = mol.ions 
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)

        func  = IONS.ComputeEnergyPotential(rho=mol.field)
        energy_ie = func.energy

        Ewald_ = ewald(rho = mol.field, ions = mol.ions, verbose = True)
        energy_ii = Ewald_.energy

        thefuncclass = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)
        func  = thefuncclass.ComputeEnergyPotential(rho=mol.field)
        energy_xc = func.energy

        # thefuncclass = FunctionalClass(type='KEDF',name='x_TF_y_vW',is_nonlocal=True)
        thefuncclass = FunctionalClass(type='KEDF',name='WT',is_nonlocal=True)
        func  = thefuncclass.ComputeEnergyPotential(rho=mol.field)
        energy_ke = func.energy

        thefuncclass = FunctionalClass(type='HARTREE',name='HARTREE',is_nonlocal=True)
        func  = thefuncclass.ComputeEnergyPotential(rho=mol.field)
        energy_h = func.energy

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
        print('energy_ie',  energy_ie * Hartree2eV, E_ie)
        print('energy_ke',  energy_ke * Hartree2eV, E_ke)
        print('energy_h ',  energy_h * Hartree2eV, E_h  )
        print('energy_ii',  energy_ii * Hartree2eV, E_ii)
        print('energy_xc',  energy_xc * Hartree2eV, E_xc)
        # print('total', energy_ie + energy_ke + energy_h + energy_ii + energy_xc, E_total/Hartree2eV)
        self.assertTrue(np.isclose(energy_ie * Hartree2eV, E_ie, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_ke * Hartree2eV, E_ke, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_h * Hartree2eV, E_h, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_ii * Hartree2eV, E_ii, rtol = 1.E-4))
        self.assertTrue(np.isclose(energy_xc * Hartree2eV, E_xc, rtol = 1.E-4))

if __name__ == "__main__":
    unittest.main()
