import unittest
import numpy as np
from pbcpy.formats.qepp import PP
from pbcpy.optimization import Optimization
from pbcpy.functionals import FunctionalClass, TotalEnergyAndPotential
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald
from pbcpy.field import DirectFieldHalf

class TestFunctional(unittest.TestCase):
    def test_optim(self):
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
        optional_kwargs = {}
        # optional_kwargs["Sigma"] = 0.0
        # optional_kwargs["x"] = 1.0
        # optional_kwargs["y"] = 1.0
        # KE = FunctionalClass(type='KEDF',name='TF',is_nonlocal=False,optional_kwargs=optional_kwargs)
        # KE = FunctionalClass(type='KEDF',name='x_TF_y_vW',is_nonlocal=False,optional_kwargs=optional_kwargs)
        KE = FunctionalClass(type='KEDF',name='WT',is_nonlocal=False,optional_kwargs=optional_kwargs)
        XC = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)
        # XC = FunctionalClass(type='XC',name='PBE',is_nonlocal=False)
        # ### load IONS and HARTREE
        HARTREE = FunctionalClass(type='HARTREE')
        nnr = mol.cell.nnr
        zerosA = np.zeros(nnr, dtype=float)
        # rho_ini = DirectFieldHalf(grid=mol.cell, griddata_F=zerosA, rank=1)
        rho_ini = DirectFieldHalf(grid=mol.cell, griddata_C=zerosA, rank=1)
        charge_total = 0.0
        for i in range(mol.ions.nat) :
            charge_total += mol.ions.Zval[mol.ions.labels[i]]

        rho_ini[:] = charge_total/mol.cell.volume

        E_v_Evaluator = TotalEnergyAndPotential(rho=rho_ini,
                                        KineticEnergyFunctional=KE,
                                        XCFunctional=XC,
                                        HARTREE=HARTREE,
                                        IONS=IONS)
        opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'TN')
        # opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'CG-HS')
        # opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'LBFGS')
        # # ### optimize!
        new_rho = opt.optimize_rho(guess_rho=rho_ini)
        # delta_rho = np.abs(new_rho - mol.field).integral()/2
        # print(delta_rho)
        # print('Energy Ewald', E_v_Evaluator.Energy(rho=rho_ini,ions=mol.ions))
        print('Calc Energy')
        Enew = E_v_Evaluator.Energy(rho=new_rho,ions=mol.ions)
        print('Calc Energy of Ref')
        Eref = E_v_Evaluator.Energy(rho=mol.field,ions=mol.ions)
        print('Energy New', Enew)
        print('Energy Ref', Eref)
        self.assertTrue(np.isclose(Enew, Eref,  rtol = 1.E-4))

if __name__ == "__main__":
    unittest.main()
