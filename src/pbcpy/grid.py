import numpy as np
from scipy import ndimage
from .base import BaseCell, DirectCell, ReciprocalCell, Coord
from .constants import LEN_CONV

class BaseGrid(BaseCell):

    '''
    Object representing a grid (Cell (lattice) plus discretization)
    extends Cell

    Attributes
    ----------
    nr : array of numbers used for discretization

    nnr : total number of subcells

    dV : volume of a subcell

    r : vectors in cartesian coordinates identifying the subcells

    s : vectors in crystal coordinates identifying the subcells

    Note:
    
    It is possible to choose between 3 different conventions for coordinates:
    - mic : 'mic'
        MIC convention.
    - mic_scaled : 'mic_scaled'
        MIC convention. Each vector i is scaled by multiplying it for nr[i]
    - normal (any other string would stick with this choice).
        NO MIC conversion.

    '''

    def __init__(self, lattice, nr, origin=np.array([0.,0.,0.]), units='Bohr', convention='mic', **kwargs):
        #print("BaseGrid __init__")
        super().__init__(lattice=lattice, origin=origin, units=units, **kwargs)
        self._nr = np.asarray(nr)
        self._nnr = nr[0] * nr[1] * nr[2]
        self._dV = self._volume / self._nnr
        #self._r = None # initialize them on request
        #self._s = None # initialize them on request

    def _calc_mask(self, ref_points):

        cutr = 1.1

        mask = np.ones(self.nr, dtype=float)
        for i in range(self.nr[0]):
            for j in range(self.nr[1]):
                for k in range(self.nr[2]):
                    for point in ref_points:
                        point = Coord(point, self)
                        # print(point)
                        dd = self.r[i, j, k].dd_mic(point)
                        if dd < cutr:
                            mask[i, j, k] = 0.
        return mask

    @property
    def r(self):
        if self._r is None:
            self._calc_grid_cart_points()
        return self._r

    @property
    def s(self):
        if self._s is None:
            self._calc_grid_crys_points()
        return self._s
    @property
    def nr(self):
        return self._nr

    @property
    def nnr(self):
        return self._nnr

    @property
    def dV(self):
        return self._dV

    # I don't see the need for these functions, so I'm gonna comment them out
    # The same can be achieved as: coord = Coord(lattice=array, cell=grid, basis="S"), instead of coord = grid.cristal_coord_array(array)
    #def crystal_coord_array(self,array):
    #    '''Returns a Coord in crystal coordinates'''
    #    if isinstance(array, (Coord)):
    #        #TODO check units
    #        return array.to_crys()
    #    else:
    #        return Coord(array, cell=self, ctype='Crystal', units=self.units)

    #def cartesian_coord_array(self,array):
    #    '''Returns a Coord in cartesian coordinates'''
    #    if isinstance(array, (Coord)):
    #        #TODO check units
    #        return array.to_cart()
    #    else:
    #        return Coord(array, cell=self, ctype='Cartesian', units=self.units)

    def square_dist_values(self,center_array=[0.,0.,0.]):
        '''Returns a ndarray with
        square distance from center_array of
        grid points in cartesian coordinates
        '''
        # assuming ctype=crystal if center_array is not a Coord object
        if isinstance(center_array, (Coord)):
            center = center_array
        else:
            center = Coord(center_array, cell=self, basis='Crystal', units=self.units)
        center_cart = center.to_cart()
        val = np.einsum('ijkl,ijkl->ijk',self.r-center_cart,self.r-center_cart)
        return val

    def dist_values(self,center_array=[0.,0.,0.]):
        '''Returns a ndarray with
        the distance from center_array of
        grid points in cartesian coordinates
        '''
        return np.sqrt(self.square_dist_values(center_array))

    def gaussianValues(self,alpha=0.1,center_array=[0.,0.,0.]):
        '''Returns a ndarray with
        the values of the gaussian
        (1/(alpha*sqrt(2pi)))*exp(-square_dist_values(center_array)/(2.0*alpha**2))
        centered on center_array
        '''
        if isinstance(alpha, (int,float,complex)):
            return (1.0/(alpha*np.sqrt(2.0*np.pi)))*np.exp(-self.square_dist_values(center_array)/(2.0*alpha**2))
        else:
            return Exception

class DirectGrid(BaseGrid,DirectCell):
    
    def __init__(self, lattice, nr, origin=np.array([0.,0.,0.]), units=None, **kwargs):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.
        """
        # internally always convert the units to Bohr
        #print("DirectGrid __init__")
        # lattice is already scaled inside the super()__init__, no need to do it here
        #lattice *= LEN_CONV[units]["Bohr"]
        super().__init__(lattice=lattice, nr=nr, origin=origin, units=units, **kwargs)
        self._r = None
        self._s = None

    def _calc_grid_crys_points(self,convention="mic"):
        if self.r is None:
            S = np.ndarray(shape=(self.nr[0], self.nr[
                           1], self.nr[2], 3), dtype=float)
            if convention == 'mic' or convention == 'mic_scaled':
                ax = []
                for i in range(3):
                    # use fftfreq function so we don't have to worry about odd or even number of points
                    dd=1
                    if convention == 'mic_scaled':
                        dd=1/self.nr[i]
                    ax.append(np.fft.fftfreq(self.nr[i],d=dd))
                    work = np.zeros(self.nr[i])
                S[:, :, :, 0], S[:, :, :, 1], S[
                    :, :, :, 2] = np.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
            else:
                s0 = np.linspace(0, 1, self.nr[0], endpoint=False)
                s1 = np.linspace(0, 1, self.nr[1], endpoint=False)
                s2 = np.linspace(0, 1, self.nr[2], endpoint=False)

                S[:,:,:,0], S[:,:,:,1], S[:,:,:,2] = np.meshgrid(s0,s1,s2,indexing='ij')
            self._s = Coord(S, cell=self, basis='Crystal')

    def _calc_grid_cart_points(self):
        if self._s is None:
            self._calc_grid_crys_points()
        self._r = self._s.to_cart()

    def get_reciprocal(self,scale=[1.,1.,1.],convention='physics'):
        """
            Returns a new ReciprocalCell, the reciprocal cell of self
            The ReciprocalCell is scaled properly to include
            the scaled (*self.nr) reciprocal grid points
            -----------------------------
            Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
            physics convention defines the reciprocal lattice to be
            exp^{i G \cdot R} = 1
            Now we have the following "crystallographer's" definition ('crystallograph')
            which comes from defining the reciprocal lattice to be
            e^{2\pi i G \cdot R} =1
            In this case bg^T = at^{-1}
            -----------------------------
            Note2: We have to use 'Bohr' units to avoid changing hbar value
        """
        # TODO define in constants module hbar value for all units allowed
        scale = np.array(scale)
        fac = 1.0
        if convention == 'physics' or convention == 'p':
            fac = 2*np.pi
        bg = fac*np.linalg.inv(self.lattice)
        bg = bg.T
        #bg = bg/LEN_CONV["Bohr"][self.units]
        reciprocal_lat = np.einsum('ij,j->ij',bg,scale)

        return ReciprocalGrid(lattice=reciprocal_lat,nr=self.nr,units=self.units)


class ReciprocalGrid(BaseGrid, ReciprocalCell):
    
    def __init__(self, lattice, nr, units=None, **kwargs):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.
        """
        # internally always convert the units to Bohr
        #print("ReciprocalGrid __init__")
        # lattice is already scaled inside the super()__init__, no need to do it here
        #lattice /= LEN_CONV[units]["Bohr"]
        super().__init__(lattice=lattice, nr=nr, origin=np.array([0.,0.,0.]), units=units, **kwargs)
        self._g = None
        self._s = None
        self._gg = None

    def _calc_grid_crys_points(self,convention="mic"):
        if self.r is None:
            S = np.ndarray(shape=(self.nr[0], self.nr[
                           1], self.nr[2], 3), dtype=float)
            if convention == 'mic' or convention == 'mic_scaled':
                ax = []
                for i in range(3):
                    # use fftfreq function so we don't have to worry about odd or even number of points
                    dd=1
                    if convention == 'mic_scaled':
                        dd=1/self.nr[i]
                    ax.append(np.fft.fftfreq(self.nr[i],d=dd))
                    work = np.zeros(self.nr[i])
                S[:, :, :, 0], S[:, :, :, 1], S[
                    :, :, :, 2] = np.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
            else:
                s0 = np.linspace(0, 1, self.nr[0], endpoint=False)
                s1 = np.linspace(0, 1, self.nr[1], endpoint=False)
                s2 = np.linspace(0, 1, self.nr[2], endpoint=False)

                S[:,:,:,0], S[:,:,:,1], S[:,:,:,2] = np.meshgrid(s0,s1,s2,indexing='ij')
            self._s = Coord(S, cell=self, basis='Crystal')

    def get_direct(self,scale=[1.,1.,1.],convention='physics'):
        """
            Returns a new DirectCell, the direct cell of self
            The DirectCell is scaled properly to include
            the scaled (*self.nr) reciprocal grid points
            -----------------------------
            Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
            physics convention defines the reciprocal lattice to be
            exp^{i G \cdot R} = 1
            Now we have the following "crystallographer's" definition ('crystallograph')
            which comes from defining the reciprocal lattice to be
            e^{2\pi i G \cdot R} =1
            In this case bg^T = at^{-1}
            -----------------------------
            Note2: We have to use 'Bohr' units to avoid changing hbar value
        """
        # TODO define in constants module hbar value for all units allowed
        scale = np.array(scale)
        fac = 1.0
        if convention == 'physics' or convention == 'p':
            fac = 1./(2*np.pi)
        at = np.linalg.inv(self.lattice.T*fac)
        #at = at*LEN_CONV["Bohr"][self.units]
        direct_lat = np.einsum('ij,j->ij',at,1./scale)
        
        return DirectGrid(lattice=direct_lat,nr=self.nr,units=self.units)


#class Grid_Space(object):
#
#    '''
#    Object representing a grid
#    (Cell (lattice) plus discretization)
#    together with its reciprocal grid
#
#    Attributes
#    ----------
#    grid : Grid
#        grid on direct space
#
#    reciprocal_grid : Grid
#        grid on reciprocal space#
#
#    nr : array of numbers used for discretization
#
#    nnr : total number of subcells
#
#    '''
#
#    def __init__(self, at, nr, origin=np.array([0.,0.,0.]), units='Bohr', convention='', reciprocal_convention='mic', conv_type='physics'):
#
#        self.grid = Grid(at, nr, origin=origin, units=units, convention=convention)
#        self.nr = self.grid.nr
#        self.nnr = self.grid.nnr
#        self.reciprocal_grid = self.grid.reciprocal_grid(reciprocal_convention=reciprocal_convention, conv_type=conv_type, scale=self.nr)
#
#    def clone(self):
#        return Grid_Space(self.grid.at,self.nr,origin=self.origin,units=self.units,reciprocal_convention=self.reciprocal_convention,conv_type=self.conv_type)
