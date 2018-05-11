from .base import Coord
from .field import ReciprocalField, DirectField
from .grid import DirectGrid, ReciprocalGrid
from .functionals import Functional
from scipy.interpolate import interp1d
import numpy as np


class Atom(object):

    def __init__(self, Z=None, Zval=None, label=None, pos=None, cell=None):

        self.Z = Z
        self.label = label
        self.Zval = Zval
        self.pos = Coord(pos, cell, basis='Cartesian')

        if Z is None:
            self.Z = z2lab.index(label)

        if label is None:
            self.label = z2lab[self.Z]

    def set_PP(self,outfile):
        '''Reads CASTEP-like recpot PP file
        Returns tuple (g, v)'''
        HARTREE2EV = 27.2113845
        BOHR2ANG   = 0.529177211
        with open(outfile,'r') as outfil:
            lines = outfil.readlines()

        for i in range(0,len(lines)):
            line = lines[i]
            if 'END COMMENT' in line:
                ibegin = i+3
            if '  1000' in line:
                iend = i
        line = " ".join([line.strip() for line in lines[ibegin:iend]])

        if '1000' in lines[iend]:
            print('Recpot pseudopotential '+outfile+' loaded')
        else:
            return Exception
        gmax = np.float(lines[ibegin-1].strip())*BOHR2ANG
        v = np.array(line.split()).astype(np.float)/HARTREE2EV/BOHR2ANG**3
        g = np.linspace(0,gmax,num=len(v))
        return g, v


    def interpolate_PP(self,g_PP,v_PP,order=None):
        '''Interpolates recpot PP
        Returns interpolation function
        Linear interpolation is the default.
        However, it can use 2nd and 3rd order interpolation
        by specifying order=n, n=1-3 in argument list.'''
        if order is None:
            order = 1
        return interp1d(g_PP,v_PP,kind=order)


    def strf(self,reciprocal_grid):
        '''
        Returns the Structure Factor associated to this ion
        '''
        a=np.exp(-1j*np.einsum('ijkl,l->ijk',reciprocal_grid.g,self.pos))
        return np.reshape(a,[reciprocal_grid.nr[0],reciprocal_grid.nr[1],reciprocal_grid.nr[2],1])


    def local_PP(self,grid,rho,outfile):
        import os.path
        if not os.path.isfile(outfile):
            print("PP file not found")
            return Exception
        reciprocal_grid = grid.get_reciprocal()
        g = reciprocal_grid.g
        q = np.sqrt(reciprocal_grid.gg)
        strf = self.strf(reciprocal_grid)
        gp, vp = self.set_PP(outfile)
        vloc_interp = self.interpolate_PP(gp, vp)
        vloc = np.zeros(np.shape(q))
        vloc[q<np.max(gp)] = vloc_interp(q[q<np.max(gp)])
        v = ReciprocalField(reciprocal_grid,griddata_3d=vloc * strf)
        vreal = DirectField(grid=grid,griddata_3d=np.real(v.ifft()))
        ereal = DirectField(grid=grid,griddata_3d=vreal*rho)
        return Functional(name='eN',energydensity=ereal, potential=vreal)

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


