# Drivers for LibXC

import numpy as np
from .atom import Atom
from .field import DirectField

def NuclearElectron(ions,density,PPs, calcType='Both'):
    '''Computes the local part of the PP
    Input: ions (coord), density (rank-0 PBCpy field), PPs (array of str)'''
    if len(PPs) != len(ions.Zval):
        raise ValueError("Incorrect number of pseudopotential files")
    if not isinstance(ions,(Atom)):
        raise AttributeError("Ions must be an array of PBCpy Atom")
    if not isinstance(density,(DirectField)):
        raise AttributeError("Density must be a PBCpy DirectField")
    NuclearElectron = ions.local_PP(grid=density.grid,rho=density,PP_file=PPs, calcType=calcType)
    NuclearElectron.name = 'Local Pseudopotential'
    return NuclearElectron

def NuclearElectronStress(ions,rho,EnergyPotential=None, PP_file=None):
    '''
    Reads and interpolates the local pseudo potential.
    INPUT: grid, rho, and path to the PP file
    OUTPUT: Functional class containing 
        - local pp in real space as potential 
        - v*rho as energy density.
    '''
    if EnergyPotential is None :
        EnergyPotential = NuclearElectron(ions, rho, PP_file, calcType='Energy')
    g=rho.grid.get_reciprocal().g
    gg=rho.grid.get_reciprocal().gg
    q = np.sqrt(gg)
    q[0, 0, 0, 0] = 1.0
    rhoG = rho.fft()
    stress = np.zeros((3, 3))
    v_deriv=ions.Get_PP_Derivative(rho.grid)
    rhoGV_q = rhoG * v_deriv / q
    for i in range(3):
        for j in range(i, 3):
            den = g[:, :, :, i]*g[:, :, :, j]
            den = den[:, :, :, np.newaxis] * rhoGV_q
            stress[i, j] = (np.einsum('ijkl->', den)).real * rho.grid.dV
            if i == j :
                stress[i, j] += EnergyPotential.energy
    stress[i, j] /= rho.grid.volume
    return stress
