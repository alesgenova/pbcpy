# Collection of local and semilocal functionals

import numpy as np
from .field import DirectField,ReciprocalField
from .grid import DirectGrid, ReciprocalGrid

def ThomasFermiPotential(self):
    '''
    The Thomas-Fermi Potential
    '''
    
    return (3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)*self**(2.0/3.0)



def ThomasFermiEnergy(self):
    '''
    The Thomas-Fermi Energy Density
    '''
    edens = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)*self**(5.0/3.0)
    return edens


def vonWeizsackerPotential(self,Sigma=0.025):
    '''
    The von Weizsacker Potential
    '''
    if not isinstance(Sigma,(np.generic,int,float)):
        print('Bad type for Sigma')
        return Exception
 
    small = np.float(1.0e-6)

    reciprocal_grid = self.grid.get_reciprocal()
    gg = reciprocal_grid.gg
    sq_dens = np.sqrt(np.real(self))
    n2_sq_dens = sq_dens.fft()*np.exp(-0.5*gg*Sigma**2)*gg
    #sq_placed = np.place(sq_dens,sq_dens<small,small)
    return DirectField(grid=self.grid,griddata_3d=0.5*np.real(n2_sq_dens.ifft())/sq_dens)

def vonWeizsackerEnergy(self):
    '''
    The von Weizsacker Energy Density
    '''
    sq_dens = np.sqrt(self)
    return DirectField(grid=self.grid,griddata_3d=0.5*np.real(np.einsum('ijkl->ijk',sq_dens.gradient()**2)))

