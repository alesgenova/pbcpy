import unittest
import numpy as np
from pbcpy.formats.qepp import PP
from pbcpy.optimization import Optimization
from pbcpy.functionals import FunctionalClass, TotalEnergyAndPotential
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald
from pbcpy.grid import DirectGrid, ReciprocalGrid
from pbcpy.field import DirectField, ReciprocalField
from pbcpy.io.vasp import  read_POSCAR

class TestFunctional(unittest.TestCase):
    def test_cluster(self):
        path_pp='tests/'
        path_pos='tests/'
        file1='Mg_lda.oe01.recpot'
        posfile='1.vasp'
        Zval = {'Mg' :2.0}
        ions = read_POSCAR(path_pos+posfile, names=['Mg'])
        ions.Zval = Zval
        lattice = ions.pos.cell.lattice
        metric = lattice * lattice.T
        gap = 1.0
        nr = np.zeros(3, dtype = 'int32')
        for i in range(3):
            nr[i] = int(np.sqrt(metric[i, i])/gap)
        print('The grid size is ', nr)
        grid = DirectGrid(lattice=lattice, nr=nr, units=None)
        zerosA = np.zeros(grid.nnr, dtype=float)
        rho_ini = DirectField(grid=grid, griddata_F=zerosA, rank=1)
        charge_total = 0.0
        for i in range(ions.nat) :
            charge_total += ions.Zval[ions.labels[i]]
        rho_ini[:] = charge_total/ions.pos.cell.volume
        print('ave',charge_total/ions.pos.cell.volume)
        optional_kwargs = {}
        optional_kwargs["PP_list"] = {'Mg': path_pp+file1}
        optional_kwargs["ions"]    = ions 
        IONS = FunctionalClass(type='IONS', optional_kwargs=optional_kwargs)
        optional_kwargs = {}
        # optional_kwargs["Sigma"] = 0.0
        # optional_kwargs["x"] = 1.0
        # optional_kwargs["y"] = 1.0
        # KE = FunctionalClass(type='KEDF',name='TF',is_nonlocal=False,optional_kwargs=optional_kwargs)
        KE = FunctionalClass(type='KEDF',name='x_TF_y_vW',is_nonlocal=False,optional_kwargs=optional_kwargs)
        # KE = FunctionalClass(type='KEDF',name='WT',is_nonlocal=False,optional_kwargs=optional_kwargs)
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
        new_rho = opt.optimize_rho(guess_rho=rho_ini)
        Enew = E_v_Evaluator.Energy(rho=new_rho,ions=ions)
        print('Energy New', Enew)

if __name__ == "__main__":
    unittest.main()
