# Collection of local and semilocal functionals

import numpy as np
from .field import DirectField,ReciprocalField
from .grid import DirectGrid, ReciprocalGrid

def ThomasFermiPotential(rho):
    '''
    The Thomas-Fermi Potential
    '''
    
    return (3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(2.0/3.0)



def ThomasFermiEnergy(rho):
    '''
    The Thomas-Fermi Energy Density
    '''
    edens = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(5.0/3.0)
    return edens


def vonWeizsackerPotential(rho,Sigma=0.025):
    '''
    The von Weizsacker Potential
    '''
    if not isinstance(Sigma,(np.generic,int,float)):
        print('Bad type for Sigma')
        return Exception
 
    small = np.float(1.0e-6)

    reciprocal_grid = rho.grid.get_reciprocal()
    gg = reciprocal_grid.gg
    sq_dens = np.sqrt(np.real(rho))
    n2_sq_dens = -sq_dens.fft()*np.exp(-gg*(Sigma)**2/4.0)*gg
    #sq_placed = np.place(sq_dens,sq_dens<small,small)
    a = 0.5*np.real(n2_sq_dens.ifft())
    return DirectField(grid=rho.grid,griddata_3d=np.divide(a,sq_dens,out=np.zeros_like(a), where=sq_dens!=0))

def vonWeizsackerEnergy(rho):
    '''
    The von Weizsacker Energy Density
    '''
    sq_dens = np.sqrt(rho)
    return DirectField(grid=rho.grid,griddata_3d=0.5*np.real(np.einsum('ijkl->ijk',sq_dens.gradient()**2)))

