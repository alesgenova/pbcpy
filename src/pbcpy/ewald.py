import numpy as np
from scipy import special as sp
import sys
from scipy.spatial.distance import cdist

from .grid import DirectGrid, ReciprocalGrid
from .field import DirectField, ReciprocalField
#from .functional_output import Functional
# from .hartree import HartreeFunctional, HartreePotentialReciprocalSpace
from itertools import product
import time

class CBspline(object):
    '''
    the Cardinal B-splines
    '''
    def __init__(self, ions = None, grid = None, order = 10, **kwargs):
        self._order = order
        self._Mn = None
        self._bm = None
        self._Barray = None
        self._PME_Qarray = None

        if ions is not None:
            self.ions      = ions
        else:
            raise AttributeError("Must pass ions to CBspline")

        if grid is not None:
            self.grid       = grid
        else:
            raise AttributeError("Must pass grid to CBspline")

    @property
    def order(self):
        return self._order

    @property
    def bm(self):
        if self._bm is None :
            self._bm = self._calc_bm()
        return self._bm

    @property
    def Barray(self):
        if self._Barray is None :
            if self._bm is None :
                self._bm = self._calc_bm()
            bm = self._bm
            array = np.einsum('i, j, k -> ijk', bm[0], bm[1], bm[2])
            self._Barray = DirectField(self.grid,griddata_3d=array,rank=1)
        return self._Barray

    @property
    def PME_Qarray(self):
        if self._PME_Qarray is None :
            self._PME_Qarray = self._calc_PME_Qarray()
        return self._PME_Qarray

    def calc_Mn(self, x, order = None):
        '''
        x -> [0, 1)
        x --> u + {0, 1, ..., order}
        u -> [0, 1), [1, 2),...,[order, order + 1)
        M_n(u) = u/(n-1)*M_(n-1)(u) + (n-u)/(n-1)*M_(n-1)(u-1)
        '''
        if not order :
            order = self.order

        Mn = np.zeros(self.order + 1)
        Mn[1] = x
        Mn[2] = 1.0 - x
        for i in range(3, order + 1):
            for j in range(0, i):
                n = i - j
                # Mn[n] = (x + n - 1) * Mn[n] + (i - (x + n - 1)) * Mn[n - 1]
                Mn[n] = (x + n - 1) * Mn[n] + (j + 1 - x) * Mn[n - 1]
                Mn[n] /= (i  - 1)
        return Mn

    def _calc_bm(self):
        nr = self.grid.nr
        Mn = self.calc_Mn(1.0)
        bm = []
        for i in range(3):
            q = 2.0 * np.pi * np.arange(nr[i]) / nr[i]
            tmp = np.exp(-1j * (self.order - 1.0) * q)
            bm.append(tmp)
            factor = np.zeros_like(bm[i])
            for k in range(1, self.order):
                factor += Mn[k] * np.exp(-1j * k * q)
            bm[i] /= factor
        return bm

    def _calc_PME_Qarray(self):
        '''
        Using the smooth particle mesh Ewald method to calculate structure factors.
        '''
        nr = self.grid.nr
        Qarray = np.zeros(nr)
        # for ion in self.ions :
            # Up = np.array(ion.pos.to_crys()) * nr
            # Mn = []
            # for i in range(3):
                # Mn.append( self.calc_Mn(Up[i] - np.floor(Up[i])) )
            # for ixyz in product(range(1, self.order + 1), repeat = 3):
                # l123 = np.mod(np.floor(Up) - ixyz, nr).astype(np.int32)
                # Qarray[tuple(l123)] += ion.Zval * Mn[0][ixyz[0]] * Mn[1][ixyz[1]] * Mn[2][ixyz[2]]

        ## For speed
        # ixyzA = np.mgrid[1:self.order + 1, 1:self.order + 1, 1:self.order + 1].reshape((3, -1))
        # l123A = np.mod(np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nr.reshape((3, 1)))
        ixyzA = np.mgrid[:self.order, :self.order, :self.order].reshape((3, -1))
        for i in range(self.ions.nat):
            Up = np.array(self.ions.pos[i].to_crys()) * nr
            Mn = []
            for j in range(3):
                Mn.append( self.calc_Mn(Up[j] - np.floor(Up[j])) )
            Mn_multi = np.einsum('i, j, k -> ijk', self.ions.Zval[self.ions.labels[i]] *  Mn[0][1:], Mn[1][1:], Mn[2][1:])
            l123A = np.mod(np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA+1, nr.reshape((3, 1)))
            Qarray[l123A[0], l123A[1], l123A[2]] += Mn_multi.reshape(-1)
        return DirectField(self.grid,griddata_3d=Qarray,rank=1)

    def get_PME_Qarray(self, i):
        '''
        Using the smooth particle mesh Ewald method to calculate structure factors.
        '''
        nr = self.grid.nr
        Qarray = np.zeros(nr)
        ixyzA = np.mgrid[:self.order, :self.order, :self.order].reshape((3, -1))
        Up = np.array(self.ions.pos[i].to_crys()) * nr
        Mn = []
        for j in range(3):
            Mn.append( self.calc_Mn(Up[j] - np.floor(Up[j])) )
        Mn_multi = np.einsum('i, j, k -> ijk', Mn[0][1:], Mn[1][1:], Mn[2][1:])
        l123A = np.mod(1 + np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA,  nr.reshape((3, 1)))
        Qarray[l123A[0], l123A[1], l123A[2]] = Mn_multi.reshape(-1)
        return DirectField(self.grid,griddata_3d=Qarray,rank=1)

class ewald(object):

    def __init__(self, precision=1.0e-8, ions=None, rho=None, verbose=False, BsplineOrder = 10, PME = False, Bspline = None):
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
        self.order = BsplineOrder

        self.usePME = PME
        if self.usePME :
            if Bspline is None :
                self.Bspline = CBspline(ions = self.ions, grid = self.rho.grid, order = self.order)
            else :
                self.Bspline = Bspline
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
        for i in np.arange(len(ions.pos)):
            charge+=ions.Zval[ions.labels[i]]
            chargeSquare+=ions.Zval[ions.labels[i]] ** 2
        
        #eta
        eta = 1.6
        NotGoodEta = True
        while NotGoodEta:
            # upbound = 2.0 * charge**2 * np.sqrt ( eta / np.pi) * sp.erfc ( np.sqrt (gmax / 4.0 / eta) )
            upbound = 4.0*np.pi*ions.nat * chargeSquare * np.sqrt ( eta / np.pi) * sp.erfc ( gmax / 2.0 * np.sqrt (1.0 / eta) )
            if upbound<precision:
                NotGoodEta = False
            else:
                eta = eta - 0.01
        return eta


    def Eewald1(self,eta, charges, positions, Rcut = 20.0):
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
        strf = ions.strf(reciprocal_grid, 0) * ions.Zval[ions.labels[0]]
        for i in np.arange(1,len(ions)):
            strf += ions.strf(reciprocal_grid, i) * ions.Zval[ions.labels[i]]
        strf_sq =np.conjugate(strf)*strf
        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0
        gg[0,0,0,0]=0.0
        First_Sum=np.real(4.0*np.pi*np.sum(strf_sq*np.exp(-gg/(4.0*eta))*invgg)) / 2.0 / rho.grid.volume
        
        # double counting term
        const=-np.sqrt(eta/np.pi)
        sum = np.float(0.0)
        for i in np.arange(len(ions.pos)):
            sum+=ions.Zval[ions.labels[i]]**2
        dc_term=const*sum 
        
        # G=0 term of local_PP - Hartree
        const=-4.0*np.pi*(1.0/(4.0*eta*rho.grid.volume)/2.0)
        sum = np.float(0.0)
        for i in np.arange(len(ions)):
            sum+=ions.Zval[ions.labels[i]]
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
                    for i in np.arange(ions.nat):
                        charges.append(ions.Zval[ions.labels[i]])
                        positions.append(ions.pos[i]-R)

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
        for i in range(ions.nat):
            dists = cdist(positions, ions.pos[i].reshape((1, 3))).reshape(-1)
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
        Z = 0.0
        for i in range(ions.nat):
            alpha += ions.alpha_mu[ions.labels[i]]
            Z +=ions.Zval[ions.labels[i]]
        alpha_z = alpha * Z / rho.grid.volume
        
        # twice Hartree term
        rhog = rho.fft()
        TwoEhart = np.sum(np.real(HartreePotentialReciprocalSpace(density=rho)*np.conjugate(rhog)))/rho.grid.volume
        vloc = ions.v
        vloc[0,0,0,0] = 0.0
        Eloc = np.real(np.sum(np.conj(rhog)*vloc))/rho.grid.volume
        return alpha_z+TwoEhart+Eloc

    def Energy_real(self):
        L=np.sqrt(np.einsum('ij->j',self.rho.grid.lattice**2))
        prec = sp.erfcinv(self.precision/3.0)
        rmax = prec / np.sqrt(self.eta)
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
                    R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),self.rho.grid.lattice)
                    for i in np.arange(self.ions.nat):
                        charges.append(self.ions.Zval[self.ions.labels[i]])
                        positions.append(self.ions.pos[i]-R)

        Esum = 0.0
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        ## for save memory
        # for item in self.ions :
            # for j in range(len(charges)):
                # rij=item.pos-positions[j]
                # dij=rij.length()
                # if dij < Rcut and dij > rtol:
                    # Esum+=charges[i]*charges[j]*sp.erfc(etaSqrt*dij)/dij
        ## for speed
        charges = np.asarray(charges)
        for i in range(self.ions.nat):
            dists = cdist(positions, self.ions.pos[i].reshape((1, 3))).reshape(-1)
            index = np.logical_and(dists < Rcut, dists > rtol)
            Esum+=self.ions.Zval[self.ions.labels[i]]*np.sum(charges[index]*sp.erfc(etaSqrt*dists[index])/ dists[index])
        Esum /= 2.0

        return Esum

    def Energy_rec(self):
        ions = self.ions
        #rec space sum
        reciprocal_grid = self.rho.grid.get_reciprocal()
        gg=reciprocal_grid.gg
        strf = ions.strf(reciprocal_grid, 0) * ions.Zval[ions.labels[0]]
        for i in np.arange(1, ions.nat):
            strf += ions.strf(reciprocal_grid, i) * ions.Zval[ions.labels[i]]
        strf_sq =np.conjugate(strf)*strf
        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0
        gg[0,0,0,0]=0.0
        energy =np.real(4.0*np.pi*np.sum(strf_sq*np.exp(-gg/(4.0*self.eta))*invgg)) / 2.0 / self.rho.grid.volume

        return energy
        
    def Energy_corr(self):
        # double counting term
        const=-np.sqrt(self.eta/np.pi)
        sum = np.float(0.0)
        for i in np.arange(self.ions.nat):
            sum+=self.ions.Zval[self.ions.labels[i]]**2
        dc_term=const*sum 
        
        # G=0 term of local_PP - Hartree
        const=-4.0*np.pi*(1.0/(4.0*self.eta*self.rho.grid.volume)/2.0)
        sum = np.float(0.0)
        for i in np.arange(self.ions.nat):
            sum+=self.ions.Zval[self.ions.labels[i]]
        gzero_limit=const*sum**2

        energy = dc_term+gzero_limit 
        
        return energy

    @property
    def energy(self):
        # gmax = self.Get_Gmax(self.rho.grid)
        # eta = self.Get_Best_eta(self.precision, gmax, self.ions)
        # self.eta = eta
        # Ewald_Energy = self.Ediv1(self.ions,self.rho)+self.Ediv2(self.precision,eta,self.ions,self.rho)
        # Ewald_Energy = self.Ediv2(self.precision,self.eta,self.ions,self.rho)
        if self.usePME :
            Ewald_Energy= self.Energy_real() + self.Energy_corr() + self.Energy_rec_PME()
        else :
            Ewald_Energy= self.Energy_real() + self.Energy_corr() + self.Energy_rec()


        if (self.verbose):
            print("Ewald sum & divergent terms in the Energy:")
            print("eta used = ", self.eta)
            print("precision used = ", self.precision)
            print("Ewald Energy = ", Ewald_Energy)
        return Ewald_Energy

    @property
    def forces(self):
        if self.usePME :
            Ewald_Forces = self.Forces_real() + self.Forces_rec_PME()
        else :
            Ewald_Forces = self.Forces_real() + self.Forces_rec()
        return Ewald_Forces

    @property
    def stress(self):

        if self.usePME :
            Ewald_Stress= self.Stress_real() + self.Stress_rec_PME()
        else :
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
                    for i in np.arange(self.ions.nat):
                        charges.append(self.ions.Zval[self.ions.labels[i]])
                        positions.append(self.ions.pos[i]-R)

        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        charges = np.asarray(charges)
        positions = np.asarray(positions)
        piSqrt = np.sqrt(np.pi)
        F_real = np.zeros((self.ions.nat, 3))
        for i in range(self.ions.nat):
            dists = cdist(positions, self.ions.pos[i].reshape((1, 3))).reshape(-1)
            index = np.logical_and(dists < Rcut, dists > rtol)
            dists *= etaSqrt
            F_real[i] = self.ions.Zval[self.ions.labels[i]] * np.einsum('ij,i->j',\
                    (np.array(self.ions.pos[i])-positions[index])*charges[index][:, np.newaxis], \
                    sp.erfc(dists[index])/ dists[index] ** 3 + \
                    2.0 / piSqrt * np.exp(-dists[index] ** 2) / dists[index] ** 2 )
        F_real *= etaSqrt ** 3

        return F_real

    def Forces_rec(self):
        reciprocal_grid = self.rho.grid.get_reciprocal()
        gg=reciprocal_grid.gg

        charges = []
        charges.append(self.ions.Zval[self.ions.labels[0]])
        strf = self.ions.strf(reciprocal_grid, 0) * self.ions.Zval[self.ions.labels[0]]
        for i in np.arange(1, self.ions.nat):
            strf += self.ions.strf(reciprocal_grid, i) * self.ions.Zval[self.ions.labels[i]]
            charges.append(self.ions.Zval[self.ions.labels[i]])

        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0
        gg[0,0,0,0]=0.0
        F_rec= np.zeros((self.ions.nat, 3))
        charges = np.asarray(charges)
        for i in range(self.ions.nat):
            Ion_strf = self.ions.strf(reciprocal_grid, i) * self.ions.Zval[self.ions.labels[i]]
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
                    for i in np.arange(self.ions.nat):
                        charges.append(self.ions.Zval[self.ions.labels[i]])
                        positions.append(self.ions.pos[i]-R)
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        charges = np.asarray(charges)
        S_real = np.zeros((3, 3))
        piSqrt = np.sqrt(np.pi)
        positions = np.asarray(positions)

        Stmp = np.zeros(6)
        for ia in range(self.ions.nat):
            dists = cdist(positions, self.ions.pos[ia].reshape((1, 3))).reshape(-1)
            index = np.logical_and(dists < Rcut, dists > rtol)
            Rijs = np.array(self.ions.pos[ia])-positions[index]

            # Rvv = np.einsum('ij, ik -> ijk', Rijs, Rijs)
            k = 0
            Rv = np.zeros((len(Rijs), 6))
            for i in range(3):
                for j in range(i, 3):
                    Rv[:, k] = Rijs[:, i] * Rijs[:, j] / dists[index] ** 2
                    k += 1

            Stmp +=self.ions.Zval[self.ions.labels[ia]]*np.einsum('i, ij->j', \
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
        strf = self.ions.strf(reciprocal_grid, 0) * self.ions.Zval[self.ions.labels[0]]
        for i in np.arange(1, self.ions.nat):
            strf += self.ions.strf(reciprocal_grid, i) * self.ions.Zval[self.ions.labels[i]]
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
                sfactor[..., k] = reciprocal_grid.g[..., i] * reciprocal_grid.g[..., j]
                sfactor[..., k] *= 2.0/gg[..., 0] * (1 + gg[:,  :,  :, 0]/(4.0 * self.eta))
                if i == j :
                    sfactor[..., k] -= 1.0
                k += 1

        gg[0,0,0,0]=0.0
        Stmp =np.einsum('ijkl, ijkl->l', strf_sq*np.exp(-gg/(4.0*self.eta))*invgg, sfactor)

        Stmp = Stmp.real * 2.0 * np.pi / self.rho.grid.volume ** 2
        # G = 0 term
        sum = np.float(0.0)
        for i in range(self.ions.nat):
            sum += self.ions.Zval[self.ions.labels[i]]
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

    def PME_Qarray_Ewald(self):
        '''
        Using the smooth particle mesh Ewald method to calculate structure factors.
        '''
        nr = self.rho.grid.nr
        Qarray = np.zeros(nr)
        Bspline = self.Bspline
        # for ion in self.ions :
            # Up = np.array(ion.pos.to_crys()) * nr
            # Mn = []
            # for i in range(3):
                # Mn.append( Bspline.calc_Mn(Up[i] - np.floor(Up[i])) )
            # for ixyz in product(range(1, self.order + 1), repeat = 3):
                # l123 = np.mod(np.floor(Up) - ixyz, nr).astype(np.int32)
                # Qarray[tuple(l123)] += ion.Zval * Mn[0][ixyz[0]] * Mn[1][ixyz[1]] * Mn[2][ixyz[2]]

        ## For speed
        ixyzA = np.mgrid[1:self.order + 1, 1:self.order + 1, 1:self.order + 1].reshape((3, -1))
        for i in range(self.ions.nat):
            Up = np.array(self.ions.pos[i].to_crys()) * nr
            Mn = []
            for j in range(3):
                Mn.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j])) )
            Mn_multi = np.einsum('i, j, k -> ijk', self.ions.Zval[self.ions.labels[i]] *  Mn[0][1:], Mn[1][1:], Mn[2][1:])
            l123A = np.mod(np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nr.reshape((3, 1)))
            Qarray[l123A[0], l123A[1], l123A[2]] += Mn_multi.reshape(-1)
        return DirectField(self.rho.grid,griddata_3d=np.reshape(Qarray,np.shape(self.rho)),rank=1)

    def Energy_rec_PME(self):
        QarrayF = self.Bspline.PME_Qarray
        # bm = self.Bspline.bm
        # method 1
        strf = QarrayF.fft()
        # b123 = np.einsum('i, j, k -> ijk', bm[0], bm[1], bm[2])
        # strf *= b123[..., np.newaxis]
        strf *= self.Bspline.Barray
        strf_sq =np.conjugate(strf)*strf
        # method 2
        # Barray = np.einsum('i, j, k -> ijk', \
                # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf_sq =np.conjugate(strf) * Barray[..., np.newaxis] * strf

        gg = self.rho.grid.get_reciprocal().gg
        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0
        gg[0,0,0,0]=0.0
        mask = self.rho.grid.get_reciprocal().mask
        mask2 = mask[..., np.newaxis]
        # energy = np.real(4.0*np.pi*np.sum(strf_sq*np.exp(-gg/(4.0*self.eta))*invgg)) / 2.0 / self.rho.grid.volume
        energy = np.sum(strf_sq[mask2]*np.exp(-gg[mask2]/(4.0*self.eta))*invgg[mask2])
        energy = 4.0* np.pi * energy.real / self.rho.grid.volume
        energy /= self.rho.grid.dV ** 2
        return energy

    def Forces_rec_PME(self):
        QarrayF = self.Bspline.PME_Qarray
        strf = QarrayF.fft()
        Bspline = self.Bspline
        Barray = Bspline.Barray
        Barray = Barray * np.conjugate(Barray)
        strf *= Barray
        # bm = Bspline.bm
        # Barray = np.einsum('i, j, k -> ijk', \
                # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf *= Barray[..., np.newaxis]
        gg = self.rho.grid.get_reciprocal().gg
        gg[0,0,0,0]=1.0
        invgg=1.0/gg
        invgg[0,0,0,0]=0.0
        gg[0,0,0,0]=0.0

        nr = self.rho.grid.nr
        strf *= np.exp(-gg/(4.0*self.eta))*invgg
        strf = strf.ifft(force_real = True)[..., 0]

        F_rec= np.zeros((self.ions.nat, 3))
        cell_inv = np.linalg.inv(self.ions.pos[0].cell.lattice)
        q_derivative = np.zeros(3)
        # for i in range(nIon):
            # Up = np.array(self.ions[i].pos.to_crys()) * nr
            # Mn = []
            # Mn_2 = []
            # for j in range(3):
                # Mn.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j])) )
                # Mn_2.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j]), order = self.order - 1) )

            # for ixyz in product(range(1, self.order + 1), repeat = 3):
                # l123 = np.mod(np.floor(Up) - ixyz, nr).astype(np.int32)
                # q_derivative[0] = (Mn_2[0][ixyz[0]] - Mn_2[0][ixyz[0]-1]) * nr[0] * Mn[1][ixyz[1]] * Mn[2][ixyz[2]]
                # q_derivative[1] = (Mn_2[1][ixyz[1]] - Mn_2[1][ixyz[1]-1]) * nr[1] * Mn[0][ixyz[0]] * Mn[2][ixyz[2]]
                # q_derivative[2] = (Mn_2[2][ixyz[2]] - Mn_2[2][ixyz[2]-1]) * nr[2] * Mn[0][ixyz[0]] * Mn[1][ixyz[1]]

                # F_rec[i] -= np.matmul(q_derivative, cell_inv) * strf[tuple(l123)]

            # F_rec[i] *= self.ions[i].Zval

        ## For speed
        ixyzA = np.mgrid[:self.order, :self.order, :self.order].reshape((3, -1))
        Q_derivativeA = np.zeros((3, self.order * self.order * self.order))
        for i in range(self.ions.nat):
            Up = np.array(self.ions.pos[i].to_crys()) * nr
            Mn = []
            Mn_2 = []
            for j in range(3):
                Mn.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j])) )
                Mn_2.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j]), order = self.order - 1) )
            Q_derivativeA[0] = nr[0] * np.einsum('i, j, k -> ijk', Mn_2[0][1:]-Mn_2[0][:-1], Mn[1][1:], Mn[2][1:]).reshape(-1)
            Q_derivativeA[1] = nr[1] * np.einsum('i, j, k -> ijk', Mn[0][1:], Mn_2[1][1:]-Mn_2[1][:-1], Mn[2][1:]).reshape(-1)
            Q_derivativeA[2] = nr[2] * np.einsum('i, j, k -> ijk', Mn[0][1:], Mn[1][1:], Mn_2[2][1:]-Mn_2[2][:-1]).reshape(-1)

            l123A = np.mod(1+np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nr.reshape((3, 1)))
            F_rec[i] -= np.sum(np.matmul(Q_derivativeA.T, cell_inv) * strf[l123A[0], l123A[1], l123A[2]][:, np.newaxis], axis=0)
            F_rec[i] *= self.ions.Zval[self.ions.labels[i]]

        F_rec *= 4.0 * np.pi / self.rho.grid.dV 

        return F_rec

    def Stress_rec_PME_full(self):
        QarrayF = self.Bspline.PME_Qarray
        # bm = self.Bspline.bm
        # method 1
        strf = QarrayF.fft()
        # b123 = np.einsum('i, j, k -> ijk', bm[0], bm[1], bm[2])
        # strf *= b123[..., np.newaxis]
        strf *= self.Bspline.Barray
        strf_sq =np.conjugate(strf)*strf
        # method 2
        # Barray = np.einsum('i, j, k -> ijk', \
                # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf_sq =np.conjugate(strf) * Barray[..., np.newaxis] * strf

        reciprocal_grid = self.rho.grid.get_reciprocal()
        gg = reciprocal_grid.gg
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
                sfactor[..., k] = reciprocal_grid.g[..., i] * reciprocal_grid.g[..., j]
                sfactor[..., k] *= 2.0/gg[..., 0] * (1 + gg[..., 0]/(4.0 * self.eta))
                if i == j :
                    sfactor[..., k] -= 1.0
                k += 1

        gg[0,0,0,0]=0.0
        Stmp =np.einsum('ijkl, ijkl->l', strf_sq*np.exp(-gg/(4.0*self.eta))*invgg, sfactor)

        Stmp = Stmp.real * 2.0 * np.pi / self.rho.grid.volume ** 2 / self.rho.grid.dV ** 2
        # G = 0 term
        sum = np.float(0.0)
        for i in range(self.ions.nat):
            sum += self.ions.Zval[self.ions.labels[i]]
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

    def Stress_rec_PME(self):
        QarrayF = self.Bspline.PME_Qarray
        # bm = self.Bspline.bm
        # method 1
        strf = QarrayF.fft()
        # b123 = np.einsum('i, j, k -> ijk', bm[0], bm[1], bm[2])
        # strf *= b123[..., np.newaxis]
        strf *= self.Bspline.Barray
        strf_sq =np.conjugate(strf)*strf
        # method 2
        # Barray = np.einsum('i, j, k -> ijk', \
                # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf_sq =np.conjugate(strf) * Barray[..., np.newaxis] * strf

        reciprocal_grid = self.rho.grid.get_reciprocal()
        gg = reciprocal_grid.gg
        mask = reciprocal_grid.mask
        mask2 = mask[..., np.newaxis]
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
                sfactor[..., k] = reciprocal_grid.g[..., i] * reciprocal_grid.g[..., j]
                sfactor[..., k] *= 2.0/gg[..., 0] * (1 + gg[..., 0]/(4.0 * self.eta))
                if i == j :
                    sfactor[..., k] -= 1.0
                Stmp[k] =2.0 * np.einsum('i, i->', strf_sq[mask2]*np.exp(-gg[mask2]/(4.0*self.eta))*invgg[mask2], sfactor[..., k][mask]).real
                k += 1

        gg[0,0,0,0]=0.0
        # Stmp =np.einsum('ijkl, ijkl->l', strf_sq*np.exp(-gg/(4.0*self.eta))*invgg, sfactor)

        Stmp = Stmp.real * 2.0 * np.pi / self.rho.grid.volume ** 2 / self.rho.grid.dV ** 2
        # G = 0 term
        sum = np.float(0.0)
        for i in range(self.ions.nat):
            sum += self.ions.Zval[self.ions.labels[i]]
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
