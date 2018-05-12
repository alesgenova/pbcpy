# Drivers for LibXC

import numpy as np
from .atom import Atom
from .field import DirectField

def NuclearElectron(ions,density,PPs):
    '''Computes the local part of the PP
    Input: ions (coord), density (rank-0 PBCpy field), PPs (array of str)'''
    if np.shape(PPs) != np.shape(ions):
        raise ValueError("Incorrect number of pseudopotential files")
    if not isinstance(ions[0],(Atom)):
        raise AttributeError("Ions must be an array of PBCpy Atom")
    if not isinstance(density,(DirectField)):
        raise AttributeError("Density must be a PBCpy DirectField")
    natoms=np.shape(ions)[0]
    NuclearElectron = ions[0].local_PP(grid=density.grid,rho=density,outfile=PPs[0])
    for i in range(1,natoms,1):
        eN_tmp = ions[i].local_PP(grid=density.grid,rho=density,outfile=PPs[i])
        NuclearElectron = NuclearElectron.sum(eN_tmp)
    NuclearElectron.name = 'Local Pseudopotential'
    return NuclearElectron






