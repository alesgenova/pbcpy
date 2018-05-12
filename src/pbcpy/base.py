import warnings
import numpy as np
from .constants import LEN_CONV, units_warning


class BaseCell(object):
    """
    Definition of the lattice of a system.

    Attributes
    ----------
    units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
        length units of the lattice vectors.
    lattice : array_like[3,3]
        matrix containing the lattice vectors of the cell (as its colums)
    omega : float
        volume of the cell in units**3

    """
    def __init__(self, lattice, origin=np.array([0.,0.,0.]), units=None, **kwargs):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct/reciprocal lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            lattice is always passed as Bohr, but we can save a preferred unit for print purposes
        """
        #print("BaseCell __init__")
        # lattice is always stored in atomic units: Bohr for direct lattices, 1/Bohr for reciprocal lattices
        self._lattice = np.asarray(lattice)
        #self.bg = np.linalg.inv(at)
        self._origin = np.asarray(origin)
        if units is not None:
            print("WARN")
            warnings.warn(units_warning, DeprecationWarning)
        self._units = None
        self._volume = np.abs(np.dot(lattice[:, 0], np.cross(lattice[:, 1], lattice[:, 2])))
        super().__init__(**kwargs)

    def __eq__(self, other):
        """
        Implement the == operator in the Cell class.

        The method is general and works even if the two cells use different
        length units.

        Parameters
        ----------
        other : Cell
            another cell object we are comparing to

        Returns
        -------
        out : Bool

        """
        if self is other:
            # if they refer to the same object, just cut to True
            return True

        # internally the lattice vectors are always saved in Bohr or 1/Bohr, no need to convert anymore
        eps = 1e-4
        #conv = LEN_CONV[other.units][self.units]
        conv = 1.0

        for ilat in range(3):
            lat0 = self.lattice[:, ilat]
            lat1 = other.lattice[:, ilat] * conv
            if not np.isclose(lat0,lat1).all():
                return False
            #overlap = np.dot(lat0, lat1) / np.dot(lat0, lat0)
            #if abs(1 - overlap) > eps:
            #    return False

        return True

    @property
    def lattice(self):
        return self._lattice

    @property
    def origin(self):
        return self._origin

    @property
    def units(self):
        #warnings.warn(units_warning, DeprecationWarning)
        return self._units

    @property
    def volume(self):
        return self._volume

    def conv(self, units):
        """
        Convert the length units of the cell, and return a new object.

        Parameters
        ----------
        units : {'Bohr', 'Angstrom', 'nm', 'm'}
            The desired length units of the Cell in output.

        Returns
        -------
        out : Cell
            New cell object with changed length unit.
        """
        # no need for a conv
        raise NotImplementedError("Cell objects are now always convertend internally to atomic units. Can only be provided when initializing.")
        #if self.units == units:
        #    return self
        #else:
        #    return Cell(at=self.at*LEN_CONV[self.units][units], units=units)

class DirectCell(BaseCell):
    
    def __init__(self, lattice, origin=np.array([0.,0.,0.]), units=None, **kwargs):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.
        """
        #print("DirectCell __init__")
        # internally always convert the units to Bohr
        #lattice *= LEN_CONV[units]["Bohr"]
        super().__init__(lattice=lattice, origin=origin, units=units, **kwargs)

    def __eq__(self, other):
        """
        Implement the == operator in the DirectCell class.
        Refer to the __eq__ method of Cell for more information.
        """
        if not isinstance(other, DirectCell):
            raise TypeError("You can only compare a DirectCell with another DirectCell")
        return super().__eq__(other)

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

        return ReciprocalCell(lattice=reciprocal_lat,units=self.units)


class ReciprocalCell(BaseCell):
    
    def __init__(self, lattice, units=None, **kwargs):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.
        """
        #print("ReciprocalCell __init__")
        # internally always convert the units to Bohr
        #lattice /= LEN_CONV[units]["Bohr"]
        super().__init__(lattice=lattice, units=units, **kwargs)
    
    def __eq__(self, other):
        """
        Implement the == operator in the ReciprocalCell class.
        Refer to the __eq__ method of Cell for more information.
        """
        if not isinstance(other, ReciprocalCell):
            raise TypeError("You can only compare a DirectCell with another DirectCell")
        return super().__eq__(other)

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
        
        return DirectCell(lattice=direct_lat,units=self.units,origin=np.array([0.,0.,0.]))
    

class Coord(np.ndarray):
    """
    Array representing coordinates in real space under periodic boundary conditions.

    Attributes
    ----------
    cell : DirectCell
        The unit cell associated to the coordinates.
    basis : {'Cartesian', 'Crystal'}
        Describes whether the array contains crystal or cartesian coordinates.

    """
    cart_names = ['Cartesian', 'Cart', 'Ca', 'R']
    crys_names = ['Crystal', 'Crys', 'Cr', 'S']

    def __new__(cls, pos, cell, basis='Cartesian'):
        """
        Parameters
        ----------
        pos : array_like[..., 3]
            Array containing a single or a set of 3D coordinates.
        cell : DirectCell
            The unit cell to be associated to the coordinates.
        basis : {'Cartesian', 'Crystal'}
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            If cell is missing, it specifies the units of the versors.
            Overridden by cell.units otherwise.

        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if not isinstance(cell, (DirectCell,)):
            raise TypeError("Coord represent coordinates in real space, cell needs to be a DirectCell", type(cell))
        
        if basis not in Coord.cart_names and basis not in Coord.crys_names:
            raise NameError("Unknown basis name: {}".format(basis))

        # Internally we always use Bohr, convert accordingly
        obj = np.asarray(pos, dtype=float).view(cls)

        if basis in Coord.cart_names:
            #obj *= LEN_CONV[cell.units]["Bohr"]
            pass
        # add the new attribute to the created instance
        obj._basis = basis
        #
        obj._cell = cell
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self._cell = getattr(obj, '_cell', None)
        self._basis = getattr(obj, '_basis', None)
        # We do not need to return anything

    def __add__(self, other):
        """
        Implement the '+' operator for the Coord class.

        Parameters
        ----------
        other : Coord | float | int | array_like
            What is to be summed to self.

        Returns
        -------
        out : Coord
            The sum of self and other.
        """
        if isinstance(other, type(self)):
        # if isinstance(other, Coord):
            if self.cell == other.cell:
                #other = other.conv(self.cell.units).to_ctype(self.ctype)
                other = other.to_basis(self.basis)
            else:
                raise Exception("Two Coord objects can only be added if they are represented in the same cell")

        return np.ndarray.__add__(self, other)

    def __mul__(self,scalar):
        """ Implement the scalar multiplication"""
        if isinstance(scalar, (int,float)):
            return np.multiply(self,scalar)
        else:
            raise TypeError("Coord can only be multiplied by a int or float scalar")

    @property
    def cell(self):
        return self._cell

    @property
    def basis(self):
        return self._basis

    def to_cart(self):
        """
        Converts the coordinates to Cartesian and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have basis='Cartesian'.
        """
        if self.basis in Coord.cart_names:
            return self
        else:
            pos = s2r(self, self.cell)
            #pos *= LEN_CONV["Bohr"][self.cell.units]
            return Coord(pos=pos, cell=self.cell, basis=Coord.cart_names[0])

    def to_crys(self):
        """
        Converts the coordinates to Crystal and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have basis='Crystal'.
        """
        if self.basis in Coord.crys_names:
            return self
        else:
            pos = r2s(self, self.cell)
            return Coord(pos=pos, cell=self.cell, basis=Coord.crys_names[0])

    def to_basis(self, basis):
        """
        Converts the coordinates to the desired basis and return a new object.

        Parameters
        ----------
        basis : {'Cartesian', 'Crystal'}
            basis to which the coordinates are converted.
        Returns
        -------
        out : Coord
            New Coord object insured to have basis=basis.
        """
        if basis in Coord.crys_names:
            return self.to_crys()
        elif basis in Coord.cart_names:
            return self.to_cart()
        else:
            raise NameError("Trying to convert to an unknown basis")

    def d_mic(self, other):
        """
        Calculate the vector connecting two Coord using the minimum image convention (MIC).

        Parameters
        ----------
        other : Coord

        Returns
        -------
        out : Coord
            shortest vector connecting self and other with the same basis as self.

        """
        ds12 = other.to_crys() - self.to_crys()
        for i in range(3):
            ds12[i] = ds12[i] - round(ds12[i])
        # dr12 = s2r(ds12, cell)
        return ds12.to_basis(self.basis)

    def dd_mic(self, other):
        """
        Calculate the distance between two Coord using the minimum image convention (MIC).

        Parameters
        ----------
        other : Coord

        Returns
        -------
        out : float
            the minimum distance between self and other from applying the MIC.

        """
        return self.d_mic(other).length()

    def length(self):
        """
        Calculate the length of a Coord array.

        Returns
        -------
        out : float
            The length of the Coord array, in the same units as self.cell.

        """
        return np.sqrt(np.dot(self.to_cart(), self.to_cart()))

    def conv(self, new_units):
        """
        Converts the units of the Coord array.

        Parameters
        ----------
        new_units : {'Bohr', 'Angstrom', 'nm', 'm'}

        Returns
        -------
        out : Coord

        """
        #new_at = self.cell.at.copy()
        #new_at *= LEN_CONV[self.cell.units][new_units]
        #return Coord(self.to_crys(), Cell(new_at, units=new_units),
        #             ctype=Coord.crys_names[0]).to_ctype(self.ctype)
        raise NotImplementedError("Coord objects are now always convertend internally to atomic units. Can only be provided when initializing.")

    def change_of_basis(self, new_cell, new_origin=np.array([0., 0., 0.])):
        """
        Perform a change of basis on the coordinates.

        Parameters
        ----------
        new_cell : Cell
            Cell representing the new coordinate system (i.e. the new basis)
        new_origin : array_like[3]
            Origin of the new coordinate system.

        Returns
        -------
        out : Coord
            Coord in the new basis.

        """
        #M = np.dot(self.cell.bg, new_cell.bg)
        #P = np.linalg.inv(M)
        #new_pos = np.dot(P, self.to_crys())
        #return Coord(new_pos, cell=new_cell)
        raise NotImplementedError("Generic change of basis non implemented yet in the Coord class")


class pbcarray(np.ndarray):
    """
    A ndarray with periodic boundary conditions when slicing (a.k.a. wrap).

    Any rank is supported.

    Examples
    --------
    2D array for semplicity of visualization, any rank should work.

    >>> dim = 3
    >>> A = np.zeros((dim,dim),dtype=int)
    >>> for i in range(dim):
    ...     A[i,i] = i+1

    >>> A = pbcarray(A)
    >>> print(A)
    [[1 0 0]
     [0 2 0]
     [0 0 3]]

    >>> print(A[-dim:,:2*dim])
    [[1 0 0 1 0 0]
     [0 2 0 0 2 0]
     [0 0 3 0 0 3]
     [1 0 0 1 0 0]
     [0 2 0 0 2 0]
     [0 0 3 0 0 3]]

    """

    def __new__(cls, pos):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pos).view(cls)
        # add the new attribute to the created instance
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

    def __getitem__(self, index):
        """
        Completely general method, works with integers, slices and ellipses,
        Periodic boundary conditions are taken into account by rolling and
        padding the array along the dimensions that need it.
        Slices with negative indexes need special treatment.

        """
        shape_ = self.shape
        rank = len(shape_)

        slices = _reconstruct_full_slices(shape_, index)
        # Now actually slice with pbc along each direction.
        newarr = self
        slice_tup = [slice(None)]*rank

        for idim, sli in enumerate(slices):
            if isinstance(sli, int):
                slice_tup[idim] = sli % shape_[idim]
            elif isinstance(sli, slice):
                roll, pad, start, stop, step = _check_slice(sli, shape_[idim])
                # If the beginning of the slice does not coincide with a grid point
                # equivalent to 0, roll the array along that axis until it does
                if roll > 0:
                    newarr = np.roll(newarr, roll, axis=idim)
                # If the span of the slice extends beyond the boundaries of the array,
                # pad the array along that axis until we have enough elements.
                if pad > 0:
                    pad_tup = [(0, 0)] * rank
                    pad_tup[idim] = (0, pad)
                    newarr = np.pad(newarr, pad_tup, mode='wrap')
                slice_tup[idim] = slice(start, stop, step)

        slice_tup = tuple(slice_tup)

        return np.ndarray.__getitem__(newarr, slice_tup)


def _reconstruct_full_slices(shape_, index):
    """
    Auxiliary function for __getitem__ to reconstruct the explicit slicing
    of the array if there are ellipsis or missing axes.

    """
    if not isinstance(index, tuple):
        index = (index,)
    slices = []
    idx_len, rank = len(index), len(shape_)

    for slice_ in index:
        if slice_ is Ellipsis:
            slices.extend([slice(None)] * (rank+1-idx_len))
        elif isinstance(slice_, slice):
            slices.append(slice_)
        elif isinstance(slice_, (int)):
            slices.append(slice_)

    sli_len = len(slices)
    if sli_len > rank:
        msg = 'too many indices for array'
        raise IndexError(msg)
    elif sli_len < rank:
        slices.extend([slice(None)]*(rank-sli_len))

    return slices


def _order_slices(dim, slices):
    """
    Order the slices span in ascending order.
    When we are slicing a pbcarray we might be rolling and padding the array
    so it's probably a good idea to make the array as small as possible
    early on.

    """
    sizes = []
    for idim, sli in slices:
        step = sli.step or 1
        start = sli.start or (0 if step > 0 else shape_[idim])
        stop = sli.stop or (shape_[idim] if step > 0 else 0)
        size = abs((max(start, stop) - min(start, stop))//step)
        sizes.append(size)

    sizes, slices = zip(*sorted(zip(sizes, slices)))

    return slices


def _check_slice(sli, dim):
    """
    Check if the current slice needs to be treated with pbc or if we can
    simply pass it to ndarray __getitem__.

    Slice is special in the following cases:
    if sli.start < 0 or > dim           # roll (and possibly pad)
    if sli.stop > dim or < 0            # roll (and possibly pad)
    if abs(sli.stop - sli.start) > 0    # pad
    """
    _roll = 0
    _pad = 0

    step = sli.step or 1
    start = (0 if step > 0 else dim) if sli.start is None else sli.start
    stop = (dim if step > 0 else 0) if sli.stop is None else sli.stop
    span = (stop - start if step > 0 else start - stop)

    if span <= 0:
        return _roll, _pad, sli.start, sli.stop, sli.step

    lower = min(start, stop)
    upper = max(start, stop)
    _start = 0 if step > 0 else span
    _stop = span if step > 0 else 0
    if span > dim:
        _pad = span - dim
        _roll = -lower % dim
    elif lower < 0 or upper > dim:
        _roll = -lower % dim
    else:
        _start = sli.start
        _stop = sli.stop

    return _roll, _pad, _start, _stop, step


def r2s(pos, cell):
    # Vectorize the code: the right most axis is where the coordinates are
    # TODO: Vectorize with einsum
    pos = np.asarray(pos)
    bg = np.linalg.inv(cell.lattice)
    xyzs = np.tensordot(bg, pos.T, axes=([-1], 0)).T
    #xyzs = np.dot(bg, pos)
    return xyzs


def s2r(pos, cell):
    # Vectorize the code: the right most axis is where the coordinates are
    # TODO: Vectorize with einsum
    pos = np.asarray(pos)
    xyzr = np.tensordot(cell.lattice, pos.T, axes=([-1], 0)).T
    return xyzr


def getrMIC(atm2, atm1, cell):
    ds12 = atm1.spos - atm2.spos
    for i in range(3):
        ds12[i] = ds12[i] - round(ds12[i])
        dr12 = s2r(ds12, cell)
    return dr12
