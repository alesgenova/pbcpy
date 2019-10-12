import unittest
import numpy as np
from pbcpy.formats.qepp import PP
from pbcpy.optimization import Optimization
from pbcpy.functionals import FunctionalClass, TotalEnergyAndPotential
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald
from pbcpy.grid import DirectGrid
from pbcpy.field import DirectField
from pbcpy.io.vasp import  read_POSCAR
import time

class TestFunctional(unittest.TestCase):
    def test_optim(self):
        print('Begin on :', time.strftime("%Y-%m-%d %H:%M:%S",  time.localtime()))
        path_pp='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/OEPP/'
        path_pos='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/POSCAR/'
        file1='Ga_lda.oe04.recpot'
        file2='As_lda.oe04.recpot'
        posfile='POSCAR_1'
        Zval = {'Ga' :3.0, 'As' :5.0}
        ions = read_POSCAR(path_pos+posfile, names=['Ga', 'As'])
        ions.Zval = Zval
        lattice = ions.pos.cell.lattice
        metric = lattice * lattice.T
        gap = 0.3
        nr = np.zeros(3, dtype = 'int32')
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i])/gap)
        print('The grid size is ', nr)
        grid = DirectGrid(lattice=lattice, nr=nr, units=None)
        # grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=False)
        zerosA = np.zeros(grid.nnr, dtype=float)
        # rho_ini = DirectField(grid=grid, griddata_F=zerosA, rank=1)
        rho_ini = DirectField(grid=grid, griddata_C=zerosA, rank=1)
        charge_total = 0.0
        for i in range(ions.nat) :
            charge_total += ions.Zval[ions.labels[i]]
        rho_ini[:] = charge_total/ions.pos.cell.volume
        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Ga': path_pp+file1,'As': path_pp+file2}
        optional_kwargs["ions"]    = ions 
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
        HARTREE = FunctionalClass(type='HARTREE')

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
        # print('Energy Ewald', E_v_Evaluator.Energy(rho=rho_ini,ions=mol.ions))
        Enew = E_v_Evaluator.Energy(rho=new_rho,ions=ions)
        print('Energy New', Enew)
        print('Finished on :', time.strftime("%Y-%m-%d %H:%M:%S",  time.localtime()))

if __name__ == "__main__":
    unittest.main()
