import numpy as np
from scipy import special as sp
import sys
from scipy.spatial.distance import cdist

from .grid import DirectGrid, ReciprocalGrid
from .field import DirectField, ReciprocalField
#from .functional_output import Functional
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

        gmax = self.Get_Gmax(self.rho.grid)
        eta = self.Get_Best_eta(self.precision, gmax, self.ions)
        self.eta = eta

            
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
        chargeSquare = 0.0 
        for i in np.arange(len(ions)):
            charge+=ions[i].Zval
            chargeSquare+=ions[i].Zval ** 2
        
        #eta
        eta = 1.6
        NotGoodEta = True
        while NotGoodEta:
            # upbound = 2.0 * charge**2 * np.sqrt ( eta / np.pi) * sp.erfc ( np.sqrt (gmax / 4.0 / eta) )
            upbound = 4.0*np.pi*len(ions) * chargeSquare * np.sqrt ( eta / np.pi) * sp.erfc ( gmax / 2.0 * np.sqrt (1.0 / eta) )
            if upbound<precision:
                NotGoodEta = False
            else:
                eta = eta - 0.01
        return eta


    def Eewald1(self,eta, charges, positions, Rcut = 12.0):
        Esum=np.float(0.0)
        for i in range(len(charges)):
            for j in range(len(charges)):
                if i!=j:
                    rij=positions[i]-positions[j]
                    dij=rij.length()
                    if dij < Rcut :
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
        # rmax = np.array([ prec / np.sqrt(eta), prec / np.sqrt(eta), prec / np.sqrt(eta)])
        rmax = prec / np.sqrt(eta)
        N=np.ceil(rmax/L)
        if self.verbose:
            print('Map of Cells = ',N)
            print('Lengths = ',rmax/L)
            print('rmax = ',rmax)
        charges = []
        positions = []
        sum = np.float(0.0)
        for ix in np.arange(-N[0],N[0]+1):
            for iy in np.arange(-N[1],N[1]+1):
                for iz in np.arange(-N[2],N[2]+1):
                    # R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),rho.grid.lattice.transpose())
                    R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),rho.grid.lattice)
                    for i in np.arange(len(ions)):
                        charges.append(ions[i].Zval)
                        positions.append(ions[i].pos-R)

        Esum = 0.0
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(eta)
        # for save memory
        # for item in ions :
            # for j in range(len(charges)):
                # rij=item.pos-positions[j]
                # dij=rij.length()
                # if dij < Rcut and dij > rtol:
                    # Esum+=charges[i]*charges[j]*sp.erfc(np.sqrt(eta)*dij)/dij
        charges = np.asarray(charges)
        for item in ions :
            dists = cdist(positions, item.pos.reshape((1, 3))).reshape(-1)
            index = np.logical_and(dists < Rcut, dists > rtol)
            Esum+=item.Zval*np.sum(charges[index]*sp.erfc(etaSqrt*dists[index])/ dists[index])
        Esum /= 2.0

        output = Esum + self.Eewald2(eta,ions,rho)
        # output = self.Eewald1(eta,charges,positions)+self.Eewald2(eta,ions,rho)
        # output = output-np.sum(np.real(HartreePotentialReciprocalSpace(density=rho)*np.conjugate(rho.fft())))/2.0/rho.grid.volume
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
        # gmax = self.Get_Gmax(self.rho.grid)
        # eta = self.Get_Best_eta(self.precision, gmax, self.ions)
        # self.eta = eta
        # Ewald_Energy = self.Ediv1(self.ions,self.rho)+self.Ediv2(self.precision,eta,self.ions,self.rho)
        Ewald_Energy = self.Ediv2(self.precision,self.eta,self.ions,self.rho)
        if (self.verbose):
            print("Ewald sum & divergent terms in the Energy:")
            print("eta used = ", self.eta)
            print("precision used = ", self.precision)
            print("Ewald Energy = ", Ewald_Energy)
        return Ewald_Energy

    @property
    def forces(self):
        Ewald_Forces = self.Forces_real() + self.Forces_rec()
        return Ewald_Forces

    @property
    def stress(self):
        Ewald_Stress= self.Stress_real() + self.Stress_rec()
        if (self.verbose):
            print('Ewald_Stress\n', Ewald_Stress)
        return Ewald_Stress


    def Forces_real(self):
        L=np.sqrt(np.einsum('ij->j',self.rho.grid.lattice**2))
        prec = sp.erfcinv(self.precision/3.0)
        rmax = prec / np.sqrt(self.eta)
        N=np.ceil(rmax/L)
        charges = []
        positions = []
        sum = np.float(0.0)
        for ix in np.arange(-N[0],N[0]+1):
            for iy in np.arange(-N[1],N[1]+1):
                for iz in np.arange(-N[2],N[2]+1):
                    R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),self.rho.grid.lattice)
                    for i in np.arange(len(self.ions)):
                        charges.append(self.ions[i].Zval)
                        positions.append(self.ions[i].pos-R)

        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        charges = np.asarray(charges)
        positions = np.asarray(positions)
        piSqrt = np.sqrt(np.pi)
        nIon = len(self.ions)
        F_real = np.zeros((nIon, 3))
        for i in range(nIon):
            dists = cdist(positions, self.ions[i].pos.reshape((1, 3))).reshape(-1)
            index = np.logical_and(dists < Rcut, dists > rtol)
            dists *= etaSqrt
            F_real[i] = self.ions[i].Zval * np.einsum('ij,i->j',\
                    (np.array(self.ions[i].pos)-positions[index])*charges[index][:, np.newaxis], \
                    sp.erfc(dists[index])/ dists[index] ** 3 + \
                    2.0 / piSqrt * np.exp(-dists[index] ** 2) / dists[index] ** 2 )
        F_real *= etaSqrt ** 3

        return F_real

    def Forces_rec(self):
        reciprocal_grid = self.rho.grid.get_reciprocal()
        gg=reciprocal_grid.gg
        strf = self.ions[0].strf(reciprocal_grid) * self.ions[0].Zval
        charges = []
        charges.append(self.ions[0].Zval)
        for i in np.arange(1,len(self.ions)):
            strf += self.ions[i].strf(reciprocal_grid) * self.ions[i].Zval
            charges.append(self.ions[i].Zval)
        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0
        gg[0,0,0,0]=0.0
        nIon = len(self.ions)
        F_rec= np.zeros((nIon, 3))
        charges = np.asarray(charges)
        for i in range(nIon):
            Ion_strf = self.ions[i].strf(reciprocal_grid) * self.ions[i].Zval
            F_rec[i] = np.einsum('ijkl,ijkl->l', reciprocal_grid.g, \
                    (Ion_strf.real * strf.imag - Ion_strf.imag * strf.real)* \
                    np.exp(-gg/(4.0*self.eta))*invgg )
        F_rec *= 4.0 * np.pi / self.rho.grid.volume

        return F_rec

    def Stress_real(self):
        L=np.sqrt(np.einsum('ij->j',self.rho.grid.lattice**2))
        prec = sp.erfcinv(self.precision/3.0)
        rmax = prec / np.sqrt(self.eta)
        N=np.ceil(rmax/L)
        charges = []
        positions = []
        sum = np.float(0.0)
        for ix in np.arange(-N[0],N[0]+1):
            for iy in np.arange(-N[1],N[1]+1):
                for iz in np.arange(-N[2],N[2]+1):
                    R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),self.rho.grid.lattice)
                    for i in np.arange(len(self.ions)):
                        charges.append(self.ions[i].Zval)
                        positions.append(self.ions[i].pos-R)
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        charges = np.asarray(charges)
        S_real = np.zeros((3, 3))
        piSqrt = np.sqrt(np.pi)
        positions = np.asarray(positions)

        Stmp = np.zeros(6)
        for ion in self.ions :
            dists = cdist(positions, ion.pos.reshape((1, 3))).reshape(-1)
            index = np.logical_and(dists < Rcut, dists > rtol)
            Rijs = np.array(ion.pos)-positions[index]

            # Rvv = np.einsum('ij, ik -> ijk', Rijs, Rijs)
            k = 0
            Rv = np.zeros((len(Rijs), 6))
            for i in range(3):
                for j in range(i, 3):
                    Rv[:, k] = Rijs[:, i] * Rijs[:, j] / dists[index] ** 2
                    k += 1

            Stmp +=ion.Zval*np.einsum('i, ij->j', \
            charges[index]*( 2 * etaSqrt / piSqrt * np.exp(-self.eta * dists[index] ** 2) + \
                    sp.erfc(etaSqrt*dists[index])/ dists[index] ), Rv)

        Stmp *= -0.5 / self.rho.grid.volume
        k = 0
        for i in range(3):
            for j in range(i, 3):
                S_real[i, j] = S_real[j, i] = Stmp[k]
                k += 1
        return S_real


    def Stress_rec(self):
        reciprocal_grid = self.rho.grid.get_reciprocal()
        gg=reciprocal_grid.gg
        strf = self.ions[0].strf(reciprocal_grid) * self.ions[0].Zval
        for i in np.arange(1,len(self.ions)):
            strf += self.ions[i].strf(reciprocal_grid) * self.ions[i].Zval
        strf_sq =np.conjugate(strf)*strf
        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0

        Stmp = np.zeros(6)
        size = list(gg.shape)
        size[-1] = 6
        sfactor = np.zeros(tuple(size))
        k = 0
        for i in range(3):
            for j in range(i, 3):
                sfactor[:, :, :, k] = reciprocal_grid.g[:, :, :, i] * reciprocal_grid.g[:, :, :, j]
                sfactor[:, :, :, k] *= 2.0/gg[:, :, :, 0] * (1 + gg[:,  :,  :, 0]/(4.0 * self.eta))
                if i == j :
                    sfactor[:, :, :, k] -= 1.0
                k += 1

        gg[0,0,0,0]=0.0
        Stmp =np.einsum('ijkl, ijkl->l', strf_sq*np.exp(-gg/(4.0*self.eta))*invgg, sfactor)

        Stmp = Stmp.real * 2.0 * np.pi / self.rho.grid.volume ** 2
        # G = 0 term
        sum = np.float(0.0)
        for ion in self.ions :
            sum += ion.Zval
        S_g0 = sum ** 2 *  4.0*np.pi*(1.0/(4.0*self.eta*self.rho.grid.volume ** 2)/2.0)
        k = 0
        S_rec = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                if i == j :
                    S_rec[i, i] = Stmp[k] + S_g0
                else :
                    S_rec[i, j] = S_rec[j, i] = Stmp[k]
                k += 1

        return S_rec
