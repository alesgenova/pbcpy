import numpy as np
from scipy import special as sp
import sys

from .grid import DirectGrid, ReciprocalGrid
from .field import DirectField, ReciprocalField
from .functionals import *
from .local_pseudopotential import NuclearElectron
from .hartree import HartreeFunctional, HartreePotentialReciprocalSpace
from .formats.qepp import PP


class ewald(object):

    def __init__(self, precision=1.0e-8, ions=None, rho=None, verbose=False):
        '''
        This computes Ewald contributions to the energy given a DirectField rho.
        INPUT: precision  float, should be bigger than the machine precision and 
                          smaller than single precision.
               ions       Atom class array.
               rho        DirectField, the electron density needed to evaluate
                          the singular parts of the energy.
               verbose    optional, wanna print stuff?
        '''


        self.precision = precision

        self.verbose = verbose

        if ions is not None:
            self.ions      = ions
        else:
            raise AttributeError("Must pass ions to Ewald")


        if rho is not None:
            self.rho       = rho
        else:
            raise AttributeError("Must pass rho to Ewald")

            
    def Get_Gmax(self,grid):
        gg = grid.get_reciprocal().gg
        gmax_x = np.sqrt(np.amax(gg[:,0,0]))
        gmax_y = np.sqrt(np.amax(gg[0,:,0]))
        gmax_z = np.sqrt(np.amax(gg[0,0,:]))
        gmax = np.amin([gmax_x,gmax_y,gmax_z])
        return gmax



    def Get_Best_eta(self,precision,gmax,ions):
        '''
        INPUT: precision, gmax & ions
        OUTPUT: eta
        '''
        
        # charge
        charge = 0.0
        for i in np.arange(len(ions)):
            charge+=ions[i].Zval
        
        #eta
        eta = 1.2
        NotGoodEta = True
        while NotGoodEta:
            upbound = 2.0 * charge**2 * np.sqrt ( eta / np.pi) * sp.erfc ( np.sqrt (gmax / 4.0 / eta) )
            if upbound<precision:
                NotGoodEta = False
            else:
                eta = eta - 0.1
        return eta


    def Eewald1(self,eta, charges, positions):
        Esum=np.float(0.0)
        for i in range(len(charges)):
            for j in range(len(charges)):
                if i!=j:
                    rij=positions[i]-positions[j]
                    dij=rij.length()
                    Esum+=charges[i]*charges[j]*sp.erfc(np.sqrt(eta)*dij)/dij
        return Esum/2.0



    def Eewald2(self,eta,ions,rho):
    
        #rec space sum
        reciprocal_grid = rho.grid.get_reciprocal()
        gg=reciprocal_grid.gg
        strf = ions[0].strf(reciprocal_grid) * ions[0].Zval
        for i in np.arange(1,len(ions)):
            strf += ions[i].strf(reciprocal_grid) * ions[i].Zval
        strf_sq =np.conjugate(strf)*strf
        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0
        gg[0,0,0,0]=0.0
        First_Sum=np.real(4.0*np.pi*np.sum(strf_sq*np.exp(-gg/(4.0*eta))*invgg)) / 2.0 / rho.grid.volume
        
        # double counting term
        const=-np.sqrt(eta/np.pi)
        sum = np.float(0.0)
        for i in np.arange(len(ions)):
            sum+=ions[i].Zval**2
        dc_term=const*sum 
        
        # G=0 term of local_PP - Hartree
        const=-4.0*np.pi*(1.0/(4.0*eta*rho.grid.volume)/2.0)
        sum = np.float(0.0)
        for i in np.arange(len(ions)):
            sum+=ions[i].Zval
        gzero_limit=const*sum**2
        
        return First_Sum+dc_term+gzero_limit


    
    def Ediv2(self,precision,eta,ions,rho):
        L=np.sqrt(np.einsum('ij->j',rho.grid.lattice**2))
        prec = sp.erfcinv(precision/3.0)
        rmax = np.array([ prec / np.sqrt(eta), prec / np.sqrt(eta), prec / np.sqrt(eta)])
        N=np.rint(rmax/L)
        if self.verbose:
            print('Map of Cells = ',N)
            print('Lengths = ',rmax/L)
        charges = []
        positions = []
        sum = np.float(0.0)
        for ix in np.arange(-N[0]+1,N[0]):
            for iy in np.arange(-N[1]+1,N[1]):
                for iz in np.arange(-N[2]+1,N[2]):
                    R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),rho.grid.lattice.transpose())
                    for i in np.arange(len(ions)):
                        charges.append(ions[i].Zval)
                        positions.append(ions[i].pos-R)
        output = self.Eewald1(eta,charges,positions)+self.Eewald2(eta,ions,rho)
        output = output-np.sum(np.real(HartreePotentialReciprocalSpace(density=rho)*np.conjugate(rho.fft())))/2.0/rho.grid.volume
        return output

    def Ediv1(self,ions,rho):
    
        # alpha Z term:
        alpha = 0.0
        for i in range(len(ions)):
            alpha += ions[i].alpha_mu
        Z = 0.0
        for i in range(len(ions)):
            Z +=ions[i].Zval
        alpha_z = alpha * Z / rho.grid.volume
        
        # twice Hartree term
        rhog = rho.fft()
        TwoEhart = np.sum(np.real(HartreePotentialReciprocalSpace(density=rho)*np.conjugate(rhog)))/rho.grid.volume
        vloc = np.zeros_like(rhog)
        for i in range(len(ions)):
            vloc+=ions[i].v
        vloc[0,0,0,0] = 0.0
        Eloc = np.real(np.sum(np.conj(rhog)*vloc))/rho.grid.volume
        return alpha_z+TwoEhart+Eloc

    @property
    def energy(self):
        gmax = self.Get_Gmax(self.rho.grid)
        eta = self.Get_Best_eta(self.precision, gmax, self.ions)
        Ewald_Energy = self.Ediv1(self.ions,self.rho)+self.Ediv2(self.precision,eta,self.ions,self.rho)
        if (self.verbose):
            print("Ewald sum & divergent terms in the Energy:")
            print("eta used = ", eta)
            print("precision used = ", precision)
            print("Ewald Energy = ", Ewald_Energy)
        return Ewald_Energy

    @property
    def forces(self):
        return Exception("Ewald forces not yet implemented")


