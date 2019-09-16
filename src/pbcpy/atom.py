from .base import Coord
from .field import ReciprocalField, DirectField
from .grid import DirectGrid, ReciprocalGrid
from .functional_output import Functional
from scipy.interpolate import interp1d, splrep, splev
import numpy as np
from .constants import LEN_CONV, ENERGY_CONV

class Atom(object):

    def __init__(self, Z=None, Zval=None, label=None, pos=None, cell=None, PP_file=None, basis='Cartesian'):
        '''
        Atom class handles atomic position, atom type and local pseudo potentials.
        '''

        self.Z = Z
        self.labels = label
        self.Zval = Zval
        # self.pos = Coord(pos, cell, basis='Cartesian')
        self.pos = Coord(pos, cell, basis=basis).to_cart()

        # private vars
        self._gp = {}       # 1D PP grid g-space 
        self._vp = {}       # PP on 1D PP grid
        self._alpha_mu = {} # G=0 of PP
        self._vlines = {}
        self._v = None        # PP for atom on 3D PW grid 
        self._vreal = None        # PP for atom on 3D real space
        self.nat = len(pos)




        # if self.PP_file is not None:
            # for f in self.PP_file :
                # gp, vp = self.set_PP(self.PP_file)
                # self._gp.append(gp)
                # self._vp.append(gp)
                # self._alpha_mu.append(self._vp[0][0])



        if Z is None:
            self.Z = []
            for item in label :
                self.Z.append(z2lab.index(item))

        if label is None:
            self.labels = []
            for item in Z :
                self.labels.append(z2lab[item])

    def set_PP(self,PP_file):
        '''Reads CASTEP-like recpot PP file
        Returns tuple (g, v)'''
        # HARTREE2EV = 27.2113845
        # BOHR2ANG   = 0.529177211
        HARTREE2EV = ENERGY_CONV['Hartree']['eV']
        BOHR2ANG   = LEN_CONV['Bohr']['Angstrom']
        with open(PP_file,'r') as outfil:
            lines = outfil.readlines()

        for i in range(0,len(lines)):
            line = lines[i]
            if 'END COMMENT' in line:
                ibegin = i+3
            if '  1000' in line:
                iend = i
        line = " ".join([line.strip() for line in lines[ibegin:iend]])

        if '1000' in lines[iend]:
            print('Recpot pseudopotential '+PP_file+' loaded')
        else:
            return Exception
        gmax = np.float(lines[ibegin-1].strip())*BOHR2ANG
        v = np.array(line.split()).astype(np.float)/HARTREE2EV/BOHR2ANG**3
        g = np.linspace(0,gmax,num=len(v))
        return g, v


    def interpolate_PP(self,g_PP,v_PP,order=3):
        '''Interpolates recpot PP
        Returns interpolation function
        Linear interpolation is the default.
        However, it can use 2nd and 3rd order interpolation
        by specifying order=n, n=1-3 in argument list.'''
        # return interp1d(g_PP,v_PP,kind=order)
        # return splrep(g_PP,v_PP,k=order)
        return splrep(g_PP,v_PP, k=order)


    def strf(self,reciprocal_grid, iatom):
        '''
        Returns the Structure Factor associated to i-th ion.
        '''
        a=np.exp(-1j*np.einsum('ijkl,l->ijk',reciprocal_grid.g,self.pos[iatom]))
        return np.reshape(a,[reciprocal_grid.nr[0],reciprocal_grid.nr[1],reciprocal_grid.nr[2],1])

    def istrf(self,reciprocal_grid, iatom):
        a=np.exp(1j*np.einsum('ijkl,l->ijk',reciprocal_grid.g,self.pos[iatom]))
        return np.reshape(a,[reciprocal_grid.nr[0],reciprocal_grid.nr[1],reciprocal_grid.nr[2],1])


    def local_PP(self,grid,rho,PP_file, calcType = 'Both'):
        '''
        Reads and interpolates the local pseudo potential.
        INPUT: grid, rho, and path to the PP file
        OUTPUT: Functional class containing 
            - local pp in real space as potential 
            - v*rho as energy density.
        '''
        if self._v is None:
            self.Get_PP_Reciprocal(grid,PP_file)
        if self._vreal is None:
            self._vreal = DirectField(grid=grid,griddata_3d=np.real(self._v.ifft()))
        ene = pot = 0
        if calcType == 'Energy' :
            ene = np.einsum('ijkl->', self._vreal * rho) * rho.grid.dV
        elif calcType == 'Potential' :
            pot = self._vreal
        else :
            ene = np.einsum('ijkl->', self._vreal * rho) * rho.grid.dV
            pot = self._vreal
        return Functional(name='eN',energy=ene, potential=pot)


    def Get_PP_Reciprocal(self,grid,PP_file):   
        import os.path

        reciprocal_grid = grid.get_reciprocal()
        g = reciprocal_grid.g
        q = np.sqrt(reciprocal_grid.gg)

        self._v = ReciprocalField(reciprocal_grid,griddata_3d=np.zeros_like(q))
        v = 1j * np.zeros_like(q)
        for key in PP_file :
            if not os.path.isfile(PP_file[key]):
                print("PP file not found")
                return Exception
            else :
                gp, vp = self.set_PP(PP_file[key])
                self._gp[key] = gp
                self._vp[key] = vp
                self._alpha_mu[key] = vp[0]
                vloc_interp = self.interpolate_PP(gp, vp)
                vloc = np.zeros(np.shape(q))
                # vloc[q<np.max(gp)] = vloc_interp(q[q<np.max(gp)])
                vloc[q<np.max(gp)] = splev(q[q<np.max(gp)], vloc_interp, der = 0)
                self._vlines[key] = vloc
                for i in range(len(self.pos)):
                    if self.labels[i] == key :
                        strf = self.strf(reciprocal_grid, i)
                        v += vloc * strf
        self._v = ReciprocalField(reciprocal_grid,griddata_3d=v)
        return "PP successfully interpolated"

    def Get_PP_Derivative(self, grid, labels = None):
        reciprocal_grid = grid.get_reciprocal()
        q = np.sqrt(reciprocal_grid.gg)
        v = 1j * np.zeros_like(q)
        if labels is None :
            labels = self._gp.keys()
        for key in labels :
            gp = self._gp[key]
            vp = self._vp[key]
            vloc_interp = self.interpolate_PP(gp, vp)
            vloc_deriv = np.zeros(np.shape(q))
            vloc_deriv[q<np.max(gp)] = splev(q[q<np.max(gp)], vloc_interp, der = 1)
            for i in range(len(self.pos)):
                if self.labels[i] == key :
                    strf = self.strf(reciprocal_grid, i)
                    v += vloc_deriv * np.conjugate(strf)
        return ReciprocalField(reciprocal_grid,griddata_3d=v)



    @property
    def v(self):
        if self._v is not None:
            return self._v
        else:
            return Exception("Must load PP first")

    @property
    def vlines(self):
        if self._vlines is not None:
            return self._vlines
        else:
            return Exception("Must load PP first")

    @property
    def alpha_mu(self):
        if self._alpha_mu is not None:
            return self._alpha_mu
        else:
            # if self._vp is not None:
                # return self._vp[0]
            # elif self.PP_file is not None:
                # self._gp, self._vp = self.set_PP(PP_file)
                # return self._vp[0]
            return Exception("Must define PP before requesting alpha_mu")
         
z2lab = ['NA', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
         'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
         'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
         'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
         'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
         'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
         'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
         'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
         'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
         'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
         'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
         'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']


