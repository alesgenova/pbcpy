# local imports
import numpy as np


def getStress(ions, rho):
    from .semilocal_xc import LDAStress
    from .local_functionals_utils import ThomasFermiStress,vonWeizsackerStress
    from .local_pseudopotential import NuclearElectronStress
    from .hartree import HartreeFunctionalStress
    from .nonlocal_functionals_utils import WTStress
    print('LDAStress\n', LDAStress(rho))
    print('ThomasFermiStress\n', ThomasFermiStress(rho))
    print('vonWeizsackerStress\n', vonWeizsackerStress(rho))
    print('NuclearElectronStress\n', NuclearElectronStress(ions, rho))
    print('HartreeFunctionalStress\n', HartreeFunctionalStress(rho))
    print('WTStress\n', WTStress(rho))

