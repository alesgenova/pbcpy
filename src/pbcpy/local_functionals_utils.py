# Collection of local and semilocal functionals

import numpy as np
from .grid_functions import Grid_Function_Base, Grid_Function, Grid_Function_Reciprocal, Grid_Space
from .grid import Grid

def ThomasFermiPotential(self):
    '''
    The Thomas-Fermi Potential
    '''
    
    return Grid_Function(self.grid_space,plot_num=self.plot_num, griddata_3d=((3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)*self.values**(2.0/3.0)))



def ThomasFermiEnergy(self):
    '''
    The Thomas-Fermi Energy Density
    '''
    edens = Grid_Function(self.grid_space,plot_num=self.plot_num, griddata_3d=((3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)*self.values**(5.0/3.0)))
    return edens


def vonWeizsackerPotential(self,SmoothingFactor=None):
    '''
    The von Weizsacker Potential
    '''
    if SmoothingFactor is not None: 
        if not isinstance(SmoothingFactor,(np.generic,int,float)):
            print('Bad type for SmoothingFactor')
            return Exception
    else:
        SmoothingFactor = 0.025
 
    sqdens_real_space = Grid_Function(self.grid_space,griddata_3d=np.sqrt(self.values))
    g2 = self.grid_space.reciprocal_grid.dist_values()**2
    sqdens_g = sqdens_real_space.fft()
    # get the gradient of sqrt(rho)
    # damp a bit the g vectors - otherwise numerics goes over the roof!
    nabla2_sqdens = Grid_Function_Reciprocal(self.grid_space,griddata_3d=g2*np.exp(-2*g2*SmoothingFactor**2)*sqdens_g.values)
    #
    v = Grid_Function(self.grid_space,plot_num=self.plot_num, griddata_3d=0.5*(nabla2_sqdens.ifft().values/sqdens_real_space.values)).real()
    return v

def vonWeizsackerEnergy(self,SmoothingFactor=None):
    '''
    The von Weizsacker Energy Density
    '''
    if SmoothingFactor is not None: 
        if not isinstance(SmoothingFactor,(np.generic,int,float)):
            print('Bad type for SmoothingFactor')
            return Exception
    else:
        SmoothingFactor = 0.025
   
    sqdens_real_space = Grid_Function(self.grid_space,griddata_3d=np.sqrt(self.values))
    g = self.grid_space.reciprocal_grid.r
    g2 = self.grid_space.reciprocal_grid.dist_values()**2
    sqdens_g = sqdens_real_space.fft()
    # get the gradient of sqrt(rho)
    # damp a bit the g vectors - otherwise numerics goes over the roof!
    nabla_sqdens_x = Grid_Function_Reciprocal(self.grid_space,griddata_3d=g[:,:,:,0]*np.exp(-g2*SmoothingFactor**2)*1j*sqdens_g.values)
    nabla_sqdens_y = Grid_Function_Reciprocal(self.grid_space,griddata_3d=g[:,:,:,1]*np.exp(-g2*SmoothingFactor**2)*1j*sqdens_g.values)
    nabla_sqdens_z = Grid_Function_Reciprocal(self.grid_space,griddata_3d=g[:,:,:,2]*np.exp(-g2*SmoothingFactor**2)*1j*sqdens_g.values)
    #
    edens = Grid_Function(self.grid_space,plot_num=self.plot_num, griddata_3d=0.5*(nabla_sqdens_x.ifft().values**2+nabla_sqdens_y.ifft().values**2+nabla_sqdens_z.ifft().values**2)).real()
    return edens





