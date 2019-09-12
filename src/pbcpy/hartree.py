# Hartree functional

import numpy as np
from .functional_output import Functional

def HartreeFunctional(density, calcType='Both'):
    gg=density.grid.get_reciprocal().gg
    rho_of_g = density.fft()
    v_h = rho_of_g.copy()
    v_h[0,0,0,0] = np.float(0.0)
    mask = gg != 0
    v_h[mask] = rho_of_g[mask]*gg[mask]**(-1)*4*np.pi
    v_h_of_r = v_h.ifft(force_real=True)
    if calcType == 'Potential' :
        e_h = 0
    else :
        e_d = v_h_of_r*density/2.0
        e_h = np.einsum('ijkl->', e_d) * density.grid.dV
        if calcType == 'Energy' :
            v_h_of_r = 0
    return Functional(name='Hartree', potential=v_h_of_r, energy=e_h)


def HartreePotentialReciprocalSpace(density):
    gg=density.grid.get_reciprocal().gg
    rho_of_g = density.fft()
    v_h = rho_of_g.copy()
    v_h[0,0,0,0] = np.float(0.0)
    mask = gg != 0
    v_h[mask] = rho_of_g[mask]*gg[mask]**(-1)*4*np.pi
    return v_h

def HartreeFunctionalStress(rho, EnergyPotential=None):
    if EnergyPotential is None :
        EnergyPotential = HartreeFunctional(rho, calcType='Energy')
    g=rho.grid.get_reciprocal().g
    gg=rho.grid.get_reciprocal().gg
    mask=rho.grid.get_reciprocal().mask

    rhoG = rho.fft() / rho.grid.volume
    gg[0, 0, 0, 0] = 1.0
    stress = np.zeros((3, 3))
    rhoG2 = rhoG * np.conjugate(rhoG) / (gg * gg)
    for i in range(3):
        for j in range(i, 3):
            den = (g[:,  :,  :, i]*g[:, :, :, j])[mask] * rhoG2[:, :, :, :][mask[:, :, :, np.newaxis]]
            Etmp = np.sum(den)
            stress[i, j] = Etmp.real* 8.0 * np.pi
            if i == j :
                stress[i, j] -= EnergyPotential.energy / rho.grid.volume
    gg[0, 0, 0, 0] = 0
    return stress
