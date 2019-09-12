import numpy as np
from pbcpy.formats.qepp import PP
from pbcpy.optimization import Optimization
from pbcpy.functionals import FunctionalClass, TotalEnergyAndPotential
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald
from pbcpy.grid import DirectGrid, ReciprocalGrid
from pbcpy.field import DirectField, ReciprocalField


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
# KE = FunctionalClass(type='KEDF',name='TF',is_nonlocal=False,optional_kwargs=optional_kwargs)
# KE = FunctionalClass(type='KEDF',name='x_TF_y_vW',is_nonlocal=False,optional_kwargs=optional_kwargs)
KE = FunctionalClass(type='KEDF',name='WT',is_nonlocal=False,optional_kwargs=optional_kwargs)
# ### load XC
# In[5]:
XC = FunctionalClass(type='XC',name='LDA',is_nonlocal=False)
# XC = FunctionalClass(type='XC',name='PBE',is_nonlocal=False)
# ### load IONS and HARTREE
# In[6]:
HARTREE = FunctionalClass(type='HARTREE')
# # load energy evaluator
# In[7]:
nnr = mol.cell.nnr
zerosA = np.zeros(nnr, dtype=float)
rho_ini = DirectField(grid=mol.cell, griddata_F=zerosA, rank=1)
charge_total = 0.0
for i in range(mol.ions.nat) :
    charge_total += mol.ions.Zval[mol.ions.labels[i]]

rho_ini[:] = charge_total/mol.cell.volume

E_v_Evaluator = TotalEnergyAndPotential(rho=rho_ini,
                                KineticEnergyFunctional=KE,
                                XCFunctional=XC,
                                HARTREE=HARTREE,
                                IONS=IONS)
# ### instance optimizer
# In[8]:
opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'TN')
# opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'CG-HS')
# opt = Optimization(EnergyEvaluator=E_v_Evaluator, optimization_method = 'LBFGS')
# # ### optimize!
# # In[9]:
# new_rho = opt.get_optimal_rho(guess_rho=rho_ini)
# new_rho = opt.optimize_rho(guess_rho=rho_ini)
rho = mol.field
new_rho = opt.optimize_rho(guess_rho=rho)
# # In[10]:
# delta_rho = np.abs(new_rho - mol.field).integral()/2
# # In[11]:
# print(delta_rho)
# In[12]:
# print('Energy Ewald', E_v_Evaluator.Energy(rho=rho_ini,ions=mol.ions))
# print('Energy New', E_v_Evaluator.Energy(rho=new_rho,ions=mol.ions))
# print('Energy Ref', E_v_Evaluator.Energy(rho=mol.field,ions=mol.ions))

from pbcpy.semilocal_xc import LDAStress
from pbcpy.local_functionals_utils import ThomasFermiStress,vonWeizsackerStress
from pbcpy.local_pseudopotential import NuclearElectronStress
from pbcpy.hartree import HartreeFunctionalStress
from pbcpy.nonlocal_functionals_utils import WTStress
print('LDAStress\n', LDAStress(rho))
print('ThomasFermiStress\n', ThomasFermiStress(rho))
optional_kwargs = {}
optional_kwargs["PP_list"] = {'Ga': path_pp+file1,'As': path_pp+file2}
print('vonWeizsackerStress\n', vonWeizsackerStress(rho))
print('NuclearElectronStress\n', NuclearElectronStress(mol.ions, rho, PP_file=optional_kwargs["PP_list"]))
print('HartreeFunctionalStress\n', HartreeFunctionalStress(rho))
# print('WTStress\n', WTStress(rho))
