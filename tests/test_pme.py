import unittest
import numpy as np

from pbcpy.functionals import FunctionalClass
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald

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
        optional_kwargs["PP_list"] = {'Ga': path_pp+file1,'As': path_pp+file2}
        optional_kwargs["ions"]    = mol.ions 
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)
        ion_pp = PP(filepp=path_ion+ionfile).read()
        func  = IONS.ComputeEnergyPotential(rho=mol.field)
        a = func.potential
        b = ion_pp.field 
        self.assertTrue(np.allclose(a,b, atol = 1.E-2))
        from pbcpy.local_pseudopotential import NuclearElectronForce, NuclearElectronForcePME
        from pbcpy.local_pseudopotential import NuclearElectronStress, NuclearElectronStressPME
        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Ga': path_pp+file1,'As': path_pp+file2}

        rho = mol.field
        IE_Energy = func.energy
        IE_Force = NuclearElectronForce(mol.ions, rho, PP_file=optional_kwargs["PP_list"])
        IE_Stress = NuclearElectronStress(mol.ions, rho, PP_file=optional_kwargs["PP_list"])

        mol = PP(filepp=path_rho+rhofile).read()
        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Ga': path_pp+file1,'As': path_pp+file2}
        optional_kwargs["ions"]    = mol.ions 
        mol.ions.usePME = True
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)
        func  = IONS.ComputeEnergyPotential(rho=mol.field)
        IE_Energy_PME = func.energy
        IE_Force_PME = NuclearElectronForcePME(mol.ions, rho, PP_file=optional_kwargs["PP_list"])
        IE_Stress_PME = NuclearElectronStressPME(mol.ions, rho, PP_file=optional_kwargs["PP_list"])

        # print(IE_Energy, IE_Energy_PME)
        self.assertTrue(np.isclose(IE_Energy, IE_Energy_PME, atol = 1.E-4))
        self.assertTrue(np.allclose(IE_Force, IE_Force_PME, atol = 1.E-4))
        self.assertTrue(np.allclose(IE_Stress, IE_Stress_PME, atol = 1.E-4))
        
    def test_ewald(self):
        Hartree2eV = 27.21138602
        print()
        print("*"*50)
        print("Testing Ewald energy")
        path_pp='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/OEPP/'
        path_rho='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/rho/'
        file1='Ga_lda.oe04.recpot'
        file2='As_lda.oe04.recpot'
        rhofile='GaAs_rho_test_1.pp'
        mol = PP(filepp=path_rho+rhofile).read()
        Ewald_ = ewald(rho = mol.field, ions = mol.ions, verbose = True)
        a = Ewald_.energy
        # print(Ewald_.forces)
        # print(Ewald_.stress)
        outfile = 'tests/Benchmarks_TOTAL_ENERGY/GaAs_test/output/GaAs_test_1.out'
        with open(outfile, 'r') as fr:
            for line in fr :
                if 'Ewald  Energy' in line :
                    b = float(line.split()[-1])
                    break
        self.assertTrue(np.isclose(a * Hartree2eV,b, atol = 1.E-4))

    def test_ewald_PME(self):
        print()
        print("*"*50)
        print("Testing particle mesh Ewald method")
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

if __name__ == "__main__":
    unittest.main()
