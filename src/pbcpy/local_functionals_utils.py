# Collection of local and semilocal functionals

import numpy as np
from .field import DirectFieldHalf,ReciprocalFieldHalf
from .functional_output import Functional
from .math_utils import TimeData, PowerInt

def ThomasFermiPotential(rho):
    '''
    The Thomas-Fermi Potential
    '''
    TimeData.Begin('TF_pot')
    factor = (3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)
    # pot = np.cbrt(rho*rho)
    # pot = factor * np.cbrt(rho * rho)
    # pot = rho * rho
    # pot = np.cbrt(pot, out = pot)
    # pot = np.multiply(factor, pot, out = pot)
    pot = rho ** (2.0/3.0) * factor
    TimeData.End('TF_pot')
    # return (3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(2.0/3.0)
    return pot



def ThomasFermiEnergy(rho):
    '''
    The Thomas-Fermi Energy
    '''
    # edens = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(5.0/3.0)
    # edens = np.cbrt(rho * rho * rho * rho * rho)
    edens  = PowerInt(rho, 5, 3)
    ene = np.einsum('ijkl->', edens) 
    ene *= (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)* rho.grid.dV
    return ene

def ThomasFermiStress(rho, EnergyPotential=None):
    '''
    The Thomas-Fermi Stress
    '''
    if EnergyPotential is None :
        EnergyPotential = TF(rho, calcType = 'Energy')
    Etmp = -2.0/3.0 * EnergyPotential.energy / rho.grid.volume
    stress = np.zeros((3, 3))
    for i in range(3):
        stress[i, i]=Etmp
    return stress

def vonWeizsackerPotential(rho,Sigma=0.025):
    '''
    The von Weizsacker Potential
    '''
    if not isinstance(Sigma,(np.generic,int,float)):
        print('Bad type for Sigma')
        return Exception
 
    gg = rho.grid.get_reciprocal().gg
    sq_dens = np.sqrt(rho)
    # n2_sq_dens = -sq_dens.fft()*np.exp(-gg*(Sigma)**2/4.0)*gg
    n2_sq_dens = sq_dens.fft()*gg
    a = n2_sq_dens.ifft()
    np.multiply(0.5, a, out = a)
    return DirectFieldHalf(grid=rho.grid,griddata_3d=np.divide(a,sq_dens,out=a))
    # return DirectFieldHalf(grid=rho.grid,griddata_3d=np.divide(a,sq_dens,out=np.zeros_like(a), where=sq_dens!=0))

def vonWeizsackerEnergy(rho, Sigma=0.025):
    '''
    The von Weizsacker Energy Density
    '''
    # sq_dens = np.sqrt(rho)
    # edens = 0.5*np.real(np.einsum('ijkl->ijk',sq_dens.gradient()**2))
    # edens = 0.5*np.real(sq_dens.gradient()**2)
    # edens = rho*vonWeizsackerPotential(rho)
    edens = vonWeizsackerPotential(rho)
    np.multiply(rho, edens, out = edens)
    ene = np.einsum('ijkl->', edens) * rho.grid.dV
    return ene

def vonWeizsackerStress(rho, EnergyPotential=None):
    '''
    The von Weizsacker Stress
    '''
    g = rho.grid.get_reciprocal().g
    rhoG = rho.fft()
    dRho_ij = []
    stress = np.zeros((3, 3))
    mask=rho.grid.get_reciprocal().mask
    mask2 = mask[..., np.newaxis]
    for i in range(3):
        dRho_ij.append((1j * g[..., i][..., np.newaxis] * rhoG).ifft())
    for i in range(3):
        for j in range(i, 3):
            Etmp = -0.25/rho.grid.volume * rho.grid.dV * np.einsum('ijkl -> ', dRho_ij[i] * dRho_ij[j]/rho)
            stress[i, j]=Etmp.real
    return stress

def vW(rho,Sigma=0.025, calcType = 'Both'):
    ene = pot = 0
    if calcType == 'Energy' :
        ene = vonWeizsackerEnergy(rho)
    elif calcType == 'Potential' :
        pot = vonWeizsackerPotential(rho,Sigma)
    else :
        pot = vonWeizsackerPotential(rho,Sigma)
        ene = np.einsum('ijkl->', rho * pot) * rho.grid.dV
        
    OutFunctional = Functional(name='vW')
    OutFunctional.potential = pot
    OutFunctional.energy= ene
    return OutFunctional

def x_TF_y_vW(rho,x=1.0,y=1.0,Sigma=0.025, calcType = 'Both'):
    xTF = TF(rho, calcType)
    yvW = vW(rho, Sigma, calcType)
    pot = x * xTF.potential + y * yvW.potential
    ene = x * xTF.energy + y * yvW.energy
    OutFunctional = Functional(name=str(x)+'_TF_'+str(y)+'_vW')
    OutFunctional.potential = pot
    OutFunctional.energy= ene
    return OutFunctional

def TF(rho, calcType = 'Both'):
    ene = pot = 0
    if calcType == 'Energy' :
        ene = ThomasFermiEnergy(rho)
    elif calcType == 'Potential' :
        pot = ThomasFermiPotential(rho)
    else :
        pot = ThomasFermiPotential(rho)
        ene = ThomasFermiEnergy(rho)
    OutFunctional = Functional(name='TF')
    OutFunctional.potential = pot
    OutFunctional.energy= ene
    return OutFunctional

