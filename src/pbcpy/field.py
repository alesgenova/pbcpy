import numpy as np
from scipy import ndimage
from .grid import Grid, Grid_Space

class ScalarField(np.ndarray):
    '''
    Extended numpy array representing a scalar field on a grid
    (Cell (lattice) plus discretization)

    Attributes
    ----------
    self : np.ndarray
        the values of the field

    grid : Grid
        Represent the domain of the function

    ndim : number of dimensions of the grid

    memo : optional string to label the field

    '''
    def __new__(cls, grid, memo="", griddata_F=None, griddata_C=None, griddata_3d=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        
        if griddata_F is None and griddata_C is None and griddata_3d is None:
            input_values = np.zeros(grid.nr)
        elif griddata_F is not None:
            input_values = np.reshape(griddata_F, grid.nr, order='F')
        elif griddata_C is not None:
            input_values = np.reshape(griddata_C, grid.nr, order='C')
        elif griddata_3d is not None:
            input_values = griddata_3d

        obj = np.asarray(input_values).view(cls)
        # add the new attribute to the created instance
        obj.grid = grid
        obj.ndim = (grid.nr > 1).sum()
        obj.memo = str(memo)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)
        self.ndim = getattr(obj, 'ndim', None)
        self.memo = getattr(obj, 'memo', None)

    def integral(self):
        ''' Returns the integral of self '''
        return np.einsum('ijk->',self)*self.grid.dV


class DirectScalarField(ScalarField):
    spl_order = 3

    def __new__(cls, grid, memo="", griddata_F=None, griddata_C=None, griddata_3d=None):
        obj = super().__new__(cls, grid, memo="", griddata_F=None, griddata_C=None, griddata_3d=None)
        obj.spl_coeffs = None
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        if obj is None: return
        super().__array_finalize__(self,obj)
        self.spl_coeffs = None

    def _calc_spline(self):
        padded_values = np.pad(self, ((self.spl_order,)), mode='wrap')
        self.spl_coeffs = ndimage.spline_filter(
            padded_values, order=self.spl_order)
        return

    def fft(self):
        ''' Implements the Discrete Fourier Transform
        - Standard FFTs -
        Compute the N(=3)-dimensional discrete Fourier Transform
        Returns a new Grid_Function_Reciprocal
        '''
        reciprocal_grid = self.grid.get_reciprocal()
        return ReciprocalScalarField(grid=reciprocal_grid, memo=self.memo, griddata_3d=np.fft.fftn(self)*self.grid.dV)

    def get_value_at_points(self, points):
        """points is in crystal coordinates"""
        if self.spl_coeffs is None:
            self._calc_spline()
        for ipol in range(3):
            # restrict crystal coordinates to [0,1)
            points[:, ipol] = (points[:, ipol] % 1) * \
                self.grid.nr[ipol] + self.spl_order
        values = ndimage.map_coordinates(self.spl_coeffs, [points[:, 0],
                                         points[:, 1], points[:, 2]],
                                         mode='wrap')
        return values

    def get_values_flatarray(self, pad=0, order='F'):
        if pad > 0:
            if self.ndim == 1:
                pad_tup = ((0,pad),(0,0),(0,0))
            elif self.ndim == 2:
                pad_tup = ((0,pad),(0,pad),(0,0))
            elif self.ndim == 3:
                pad_tup = ((0,pad),(0,pad),(0,pad))
            vals = np.pad(self, (0,pad), mode='wrap')
        else:
            vals = np.asarray(self)
        nr = vals.shape
        nnr = 1
        for n in nr:
            nnr *= n
        #nnr = nr[0] * nr[1] * nr[2]
        #print(nr, nnr)
        return np.reshape(vals, nnr, order=order)

    def get_3dinterpolation(self, nr_new):
        """
        Interpolates the values of the function on a cell with a different number
        of points, and returns a new Grid_Function_Base object.
        """
        if self.spl_coeffs is None:
            self._calc_spline()
        x = np.linspace(0, 1, nr_new[0], endpoint=False) * \
            self.grid.nr[0] + self.spl_order
        y = np.linspace(0, 1, nr_new[1], endpoint=False) * \
            self.grid.nr[1] + self.spl_order
        z = np.linspace(0, 1, nr_new[2], endpoint=False) * \
            self.grid.nr[2] + self.spl_order
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        new_values = ndimage.map_coordinates(
            self.spl_coeffs, [X, Y, Z], mode='wrap')
        new_grid = Grid(self.grid.lattice, nr_new, units=self.grid.units)
        return DirectScalarField(new_grid, self.memo, griddata_3d=new_values)

    def get_plotcut(self, x0, r0, r1=None, r2=None, nr=10):
        """
        general routine to get the arbitrary cuts of a Grid_Function_Base object in 1,2,
        or 3 dimensions. spline interpolation will be used.
            x0 = origin of the cut
            r0 = first vector (always required)
            r1 = second vector (required for 2D and 3D cuts)
            r2 = third vector (required for 3D cuts)
            nr[i] = number points to discretize each direction ; i = 0,1,2
        x0, r0, r1, r2 are all in crystal coordinates
        """

        ndim = 1

        x0 = x0.to_crys()
        r0 = r0.to_crys()
        if r1 is not None:
            r1 = r1.to_crys()
            ndim += 1
            if r2 is not None:
                r2 = r2.to_crys()
                ndim += 1
        nrx = np.ones(3, dtype=int)
        if isinstance(nr, (int, float)):
            nrx[0:ndim] = nr
        elif isinstance(nr, (np.ndarray, list, tuple)):
            nrx[0:ndim] = np.asarray(nr, dtype=int)

        dr = np.zeros((3, 3), dtype=float)
        dr[0, :] = (r0) / nrx[0]
        if ndim > 1:
            dr[1, :] = (r1) / nrx[1]
            if ndim == 3:
                dr[2, :] = (r2) / nrx[2]
        points = np.zeros((nrx[0], nrx[1], nrx[2], 3))
        axis = []
        for ipol in range(3):
            axis.append(np.zeros((nrx[ipol], 3)))
            for ir in range(nrx[ipol]):
                axis[ipol][ir, :] = ir * dr[ipol]

        for i in range(nrx[0]):
            for j in range(nrx[1]):
                for k in range(nrx[2]):
                    points[i, j, k, :] = x0 + axis[0][i, :] + \
                        axis[1][j, :] + axis[2][k, :]

        a, b, c, d = points.shape
        points = points.reshape((a * b * c, 3))

        values = self.get_value_at_points(points)

        # generate a new grid (possibly 1D/2D/3D)
        origin = x0.to_cart()
        at = np.identity(3)
        v0 = r0.to_cart()
        v1 = np.zeros(3)
        v2 = np.zeros(3)
        # We still need to define 3 lattice vectors even if the plot is in 1D/2D
        # Here we ensure the 'ficticious' vectors are orthonormal to the actual ones
        # so that units of length/area are correct.
        if ndim == 1:
            for i in range(3):
                if abs(v0[i]) > 1e-4:
                    j = i - 1
                    v1[j] = v0[i]
                    v1[i] = -v0[j]
                    v1 = v1 / np.sqrt(np.dot(v1,v1))
                    break
            v2 = np.cross(v0,v1)
            v2 = v2 / np.sqrt(np.dot(v2,v2))
        elif ndim  == 2:
            v1 = r1.to_cart()
            v2 = np.cross(v0,v1)
            v2 = v2/ np.sqrt(np.dot(v2,v2))
        elif ndim == 3:
            v1 = r1.to_cart()
            v2 = r2.to_cart()
        at[:,0] = v0
        at[:,1] = v1
        at[:,2] = v2

        cut_grid = Grid(lattice=at, nr=nrx, origin=origin, units=x0.cell.units)

        if ndim == 1:
            values = values.reshape((a))
        elif ndim == 2:
            values = values.reshape((a, b))
        elif ndim == 3:
            values = values.reshape((a, b, c))

        return DirectScalarField(grid=cut_grid, memo=self.memo, griddata_3d=values)

class ReciprocalScalarField(ScalarField):

    def __new__(cls, grid, memo="", griddata_F=None, griddata_C=None, griddata_3d=None):
        obj = super().__new__(cls, grid, memo="", griddata_F=None, griddata_C=None, griddata_3d=None)
        obj.spl_coeffs = None
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        if obj is None: return
        super().__array_finalize__(self,obj)
        self.spl_coeffs = None

    def ifft(self):
        '''
        Implements the Inverse Discrete Fourier Transform
        - Standard FFTs -
        Compute the N(=3)-dimensional inverse discrete Fourier Transform
        Returns a new Grid_Function
        '''
        direct_grid = self.grid.get_direct()
        return DirectScalarField(grid=direct_grid, memo=self.memo, griddata_3d=np.fft.ifftn(self)/self.grid.dV)
