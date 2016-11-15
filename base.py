import numpy as np
from .constants import LEN_CONV


class Cell(object):

    def __init__(self, at, units='Bohr'):
        self.at = np.asarray(at)
        self.units = units
        self.bg = np.linalg.inv(at)
        self.omega = np.dot(at[:, 0], np.cross(at[:, 1], at[:, 2]))
        # self.alat = np.sqrt(np.dot(at[:][0], at[:][0]))

    def __eq__(self, other):
        """
        method to expose == operator to Cell object.
        It is general and works for cells with different units.
        """
        if self is other:
            # if they refer to the same object, just cut to True
            return True

        common_unit = 'Bohr'  # arbitrary
        eps = 1e-3
        conv0 = LEN_CONV[self.units][common_unit]
        conv1 = LEN_CONV[other.units][common_unit]

        for ilat in range(3):
            lat0 = self.at[:, ilat] * conv0
            lat1 = other.at[:, ilat] * conv1
            overlap = np.dot(lat0, lat1) / np.dot(lat0, lat0)
            if abs(1 - overlap) > eps:
                return False

        return True

    def set_grid(self, nr):

        pass


class Coord(np.ndarray):
    cart_names = ['Cartesian', 'Cart', 'Ca', 'R']
    crys_names = ['Crystal', 'Crys', 'Cr', 'S']

    def __new__(cls, pos, cell, ctype='Cartesian'):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pos, dtype=float).view(cls)
        # add the new attribute to the created instance
        obj.cell = cell
        obj.ctype = ctype
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.cell = getattr(obj, 'cell', None)
        self.ctype = getattr(obj, 'ctype', None)
        # We do not need to return anything

    def __add__(self, other):
        """
        sum the coordinates of two Coord objects (can seamlessly mix Cris and
        Cart) and return a new Coord with the same ctype as the first Coord
        object (self).
        If ctype_out is specified, the output Coord object is converted to it.
        """
        if isinstance(other, Coord):
            other = other.to_ctype(self.ctype)

        return np.ndarray.__add__(self, other)

    def add(self, other, ctype_out=None):
        """
        sum the coordinates of two Coord objects (can seamlessly mix Cris and
        Cart) and return a new Coord with the same ctype as the first Coord
        object (self).
        If ctype_out is specified, the output Coord object is converted to it.
        """
        if ctype_out is None:
            return self + other.to_ctype(self.ctype)
        else:
            return self.to_ctype(ctype_out) + other.to_ctype(ctype_out)

    def to_cart(self):
        if self.ctype in Coord.cart_names:
            return self
        else:
            pos = s2r(self, self.cell)
            return Coord(pos=pos, cell=self.cell, ctype=Coord.cart_names[0])

    def to_crys(self):
        if self.ctype in Coord.crys_names:
            return self
        else:
            pos = r2s(self, self.cell)
            return Coord(pos=pos, cell=self.cell, ctype=Coord.crys_names[0])

    def to_ctype(self, ctype):
        if ctype in Coord.crys_names:
            return self.to_crys()
        elif ctype in Coord.cart_names:
            return self.to_cart()

    def d_mic(self, other):
        ds12 = other.to_crys() - self.to_crys()
        for i in range(3):
            ds12[i] = ds12[i] - round(ds12[i])
        # dr12 = s2r(ds12, cell)
        return ds12.to_ctype(self.ctype)

    def dd_mic(self, other):
        return self.d_mic(other).lenght()

    def lenght(self):
        return np.sqrt(np.dot(self.to_cart(), self.to_cart()))

    def conv(self, new_units):
        '''
        returns a Coord obj with new_units as its units.
        This is accomplished by first changing the self.cell and then
        changing the array coordinates accordingly.
        '''
        # new_at = self.cell.at.copy()
        new_at = self.cell.at.copy()
        new_at *= LEN_CONV[self.cell.units][new_units]
        # new_cell = Cell(new_at,units=new_units)
        return Coord(self.to_crys(), Cell(new_at, units=new_units),
                     ctype=Coord.crys_names[0]).to_ctype(self.ctype)

    def same_cell_as(self, other):
        """
        checks if the unit cell of two Coord objects is the same,
        regardless of the units.
        """
        pass


class pbcarray(np.ndarray):

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

        # self.cell = getattr(obj, 'cell', None)
        # self.ctype = getattr(obj, 'ctype', None)
        # We do not need to return anything

    def __getitem__(self, sli_in):

        # the slice in input can as well be an integer, or a tuple of
        # integers/slices with length smaller than the rank of the array.
        print(sli_in)
        noneslice = slice(None, None, None)
        shp = self.shape
        rank = len(shp)
        noneslicelist = [noneslice] * rank

        sli = []

        try:
            ninput = len(sli_in)
            if ninput > rank:
                print('Array dim mismatch')
                return

        except:

            ninput = 1
            sli_in = (sli_in,)

        sli = noneslicelist.copy()

        for idim in range(ninput):
            sli[idim] = sli_in[idim]

        newarr = np.asarray(self)

        for isl, sl in enumerate(sli):

            if isinstance(sl, int):
                slice_tup = noneslicelist.copy()
                slice_tup[isl] = sl
                slice_tup = tuple(slice_tup)
                newarr = newarr[slice_tup]

            elif isinstance(sl, slice):

                start = sl.start
                stop = sl.stop
                step = sl.step

                step_ = sl.step

                if step is None:
                    step = 1

                if step > 0:

                    if start is None:
                        start = 0

                    if stop is None:
                        stop = shp[isl]

                elif step < 0:

                    if start is None:
                        start = shp[isl]

                    if stop is None:
                        stop = 0

                lower = min(start, stop)
                upper = max(start, stop)
                span = upper - lower

                roll = 0

                if lower % shp[isl] != 0:
                    roll = -lower % shp[isl]
                    newarr = np.roll(newarr, roll, axis=isl)

                if span > shp[isl]:
                    pad_tup = [(0, 0)] * rank
                    pad_tup[isl] = (0, span - shp[isl])
                    newarr = np.pad(newarr, pad_tup, mode='wrap')

                if True:
                    slice_tup = noneslicelist.copy()
                    if step < 0:
                        slice_tup[isl] = slice(
                            start + roll, stop + roll, step_)
                    else:
                        slice_tup[isl] = slice(None, span, step)

                    slice_tup = tuple(slice_tup)
                    newarr = newarr[slice_tup]

        return newarr


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
