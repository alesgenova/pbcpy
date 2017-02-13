import numpy as np
from .constants import LEN_CONV


class Cell(object):
    """
    Definition of the lattice of a system.

    Attributes
    ----------
    units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
        length units of the lattice vectors.
    at : array_like[3,3]
        matrix containing the direct lattice vectors (as its colums)
    bg : array_like[3,3]
        matrix containing the reciprocal lattice vectors (i.e. inverse of at)
    omega : float
        volume of the cell in units**3

    """
    def __init__(self, at, origin=np.array([0.,0.,0.]), units='Bohr'):
        """
        Parameters
        ----------
        at : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            length units of the lattice vectors.

        """
        self.at = np.asarray(at)
        self.origin = np.asarray(origin)
        self.units = units
        self.bg = np.linalg.inv(at)
        self.omega = np.dot(at[:, 0], np.cross(at[:, 1], at[:, 2]))
        # self.alat = np.sqrt(np.dot(at[:][0], at[:][0]))

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

        eps = 1e-4
        conv = LEN_CONV[other.units][self.units]

        for ilat in range(3):
            lat0 = self.at[:, ilat]
            lat1 = other.at[:, ilat] * conv
            overlap = np.dot(lat0, lat1) / np.dot(lat0, lat0)
            if abs(1 - overlap) > eps:
                return False

        return True

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
        if self.units == units:
            return self
        else:
            return Cell(at=self.at*LEN_CONV[self.units][units], units=units)


class Coord(np.ndarray):
    """
    Array representing coordinates in periodic boundary conditions.

    Attributes
    ----------
    cell : Cell
        The unit cell associated to the coordinates.
    ctype : {'Cartesian', 'Crystal'}
        Describes whether the array contains crystal or cartesian coordinates.

    """
    cart_names = ['Cartesian', 'Cart', 'Ca', 'R']
    crys_names = ['Crystal', 'Crys', 'Cr', 'S']

    def __new__(cls, pos, cell=None, ctype='Cartesian', units='Bohr'):
        """
        Parameters
        ----------
        pos : array_like[..., 3]
            Array containing a single or a set of 3D coordinates.
        cell : Cell
            The unit cell to be associated to the coordinates.
        ctype : {'Cartesian', 'Crystal'}
            matrix containing the direct lattice vectors (as its colums)
        units : {'Bohr', 'Angstrom', 'nm', 'm'}, optional
            If cell is missing, it specifies the units of the versors.
            Overridden by cell.units otherwise.

        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pos, dtype=float).view(cls)
        # add the new attribute to the created instance
        if cell is None:
            # If no cell in input, coordinates are purely cartesian,
            # i.e. the lattice vectors are three orthogonal versors i, j, k.
            obj.cell = Cell(np.identity(3), units=units)
        else:
            obj.cell = cell
        obj.ctype = ctype
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.cell = getattr(obj, 'cell', None)
        self.ctype = getattr(obj, 'ctype', None)
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
                other = other.conv(self.cell.units).to_ctype(self.ctype)
            else:
                return Exception

        return np.ndarray.__add__(self, other)

    def to_cart(self):
        """
        Converts the coordinates to Cartesian and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have ctype='Cartesian'.
        """
        if self.ctype in Coord.cart_names:
            return self
        else:
            pos = s2r(self, self.cell)
            return Coord(pos=pos, cell=self.cell, ctype=Coord.cart_names[0])

    def to_crys(self):
        """
        Converts the coordinates to Crystal and return a new Coord object.

        Returns
        -------
        out : Coord
            New Coord object insured to have ctype='Crystal'.
        """
        if self.ctype in Coord.crys_names:
            return self
        else:
            pos = r2s(self, self.cell)
            return Coord(pos=pos, cell=self.cell, ctype=Coord.crys_names[0])

    def to_ctype(self, ctype):
        """
        Converts the coordinates to the desired ctype and return a new object.

        Parameters
        ----------
        ctype : {'Cartesian', 'Crystal'}
            ctype to which the coordinates are converted.
        Returns
        -------
        out : Coord
            New Coord object insured to have ctype=ctype.
        """
        if ctype in Coord.crys_names:
            return self.to_crys()
        elif ctype in Coord.cart_names:
            return self.to_cart()

    def d_mic(self, other):
        """
        Calculate the vector connecting two Coord using the minimum image convention (MIC).

        Parameters
        ----------
        other : Coord

        Returns
        -------
        out : Coord
            shortest vector connecting self and other with the same ctype as self.

        """
        ds12 = other.to_crys() - self.to_crys()
        for i in range(3):
            ds12[i] = ds12[i] - round(ds12[i])
        # dr12 = s2r(ds12, cell)
        return ds12.to_ctype(self.ctype)

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
        return self.d_mic(other).lenght()

    def length(self):
        """
        Calculate the legth of a Coord array.

        Returns
        -------
        out : float
            The lenght of the Coord array, in the same units as self.cell.

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
        # new_at = self.cell.at.copy()
        new_at = self.cell.at.copy()
        new_at *= LEN_CONV[self.cell.units][new_units]
        # new_cell = Cell(new_at,units=new_units)
        return Coord(self.to_crys(), Cell(new_at, units=new_units),
                     ctype=Coord.crys_names[0]).to_ctype(self.ctype)

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
        M = np.dot(self.cell.bg, new_cell.bg)
        P = np.linalg.inv(M)
        new_pos = np.dot(P, self.to_crys())
        return Coord(new_pos, cell=new_cell)


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
    pos = np.asarray(pos)
    xyzs = np.tensordot(cell.bg, pos.T, axes=([-1], 0)).T
    # xyzs = np.dot(cell.bg, pos)
    return xyzs


def s2r(pos, cell):
    # Vectorize the code: the right most axis is where the coordinates are
    pos = np.asarray(pos)
    xyzr = np.tensordot(cell.at, pos.T, axes=([-1], 0)).T
    return xyzr


def getrMIC(atm2, atm1, cell):
    ds12 = atm1.spos - atm2.spos
    for i in range(3):
        ds12[i] = ds12[i] - round(ds12[i])
        dr12 = s2r(ds12, cell)
    return dr12
