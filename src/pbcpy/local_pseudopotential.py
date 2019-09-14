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
    if EnergyPotential is None :
        EnergyPotential = NuclearElectron(ions, rho, PP_file, calcType='Energy')
    reciprocal_grid=rho.grid.get_reciprocal()
    g= reciprocal_grid.g
    gg= reciprocal_grid.gg
    mask = reciprocal_grid.mask
    mask2 = mask[..., np.newaxis]
    q = np.sqrt(gg)
    q[0, 0, 0, 0] = 1.0
    rhoG = rho.fft()
    stress = np.zeros((3, 3))
    v_deriv=ions.Get_PP_Derivative(rho.grid)
    rhoGV_q = rhoG * v_deriv / q
    for i in range(3):
        for j in range(i, 3):
            # den = (g[..., i]*g[..., j])[..., np.newaxis] * rhoGV_q
            # stress[i, j] = (np.einsum('ijkl->', den)).real / rho.grid.volume
            den = (g[..., i][mask]*g[..., j][mask]) * rhoGV_q[mask2]
            stress[i, j] = (np.einsum('i->', den)).real / rho.grid.volume*2.0
            if i == j :
                stress[i, j] += EnergyPotential.energy
    stress /= rho.grid.volume
    return stress

def NuclearElectronForce(ions,rho,PP_file=None):
    rhoG = rho.fft()
    reciprocal_grid = rho.grid.get_reciprocal()
    g = reciprocal_grid.g
    Forces= np.zeros((ions.nat, 3))
    mask = reciprocal_grid.mask
    mask2 = mask[..., np.newaxis]
    # for i in range(ions.nat):
        # strf = ions.istrf(reciprocal_grid, i)
        # Forces[i] = np.einsum('ijkl,ijkl->l', reciprocal_grid.g, \
                # ions.vlines[ions.labels[i]]* (rhoG * strf).imag)
    # Forces /= rho.grid.volume
    for i in range(ions.nat):
        strf = ions.istrf(reciprocal_grid, i)
        den = ions.vlines[ions.labels[i]][mask2]* (rhoG[mask2] * strf[mask2]).imag
        for j in range(3):
            Forces[i, j] = np.einsum('i, i->', reciprocal_grid.g[..., j][mask], den)
    Forces *= 2.0/rho.grid.volume
    return Forces
