import warnings
import numpy as np
from scipy import ndimage
from .grid import DirectGrid, ReciprocalGrid
from .constants import LEN_CONV


class BaseField(np.ndarray):
    '''
    Extended numpy array representing a field on a grid
    (Cell (lattice) plus discretization)

    Attributes
    ----------
    self : np.ndarray
        the values of the field

    grid : Grid
        Represent the domain of the function

    span : number of directions for which we have more than 1 point
            e.g.: for np.zeros((5,5,1)) -> ndim = 3, span = 2

    rank : rank of the field. default = 1, i.e. scalar field

    memo : optional string to label the field

    '''
    def __new__(cls, grid, memo="", rank=None, griddata_F=None, griddata_C=None, griddata_3d=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        rank = 1
        nr = *grid.nr, rank
        if griddata_3d is not None:
           a=np.shape(np.shape(griddata_3d))[0] 
           if a == 4:
               rank = np.shape(griddata_3d)[3]
               nr = *grid.nr, rank
           #elif a == 3:
           #    rank = 1 
           #    nr = grid.nr


        if griddata_F is None and griddata_C is None and griddata_3d is None:
            input_values = np.zeros(nr)
        elif griddata_F is not None:
            input_values = np.reshape(griddata_F, nr, order='F')
        elif griddata_C is not None:
            input_values = np.reshape(griddata_C, nr, order='C')
        elif griddata_3d is not None:
            # input_values = griddata_3d
            input_values = np.reshape(griddata_3d, nr)

        obj = np.asarray(input_values).view(cls)
        # add the new attribute to the created instance
        obj.grid = grid
        obj.span = (grid.nr > 1).sum()
        obj.rank = rank
        obj.memo = str(memo)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        #print(type(self))
        #print(type(obj))
        #print(type(args[0]))
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)
        self.span = getattr(obj, 'span', None)
        self.memo = getattr(obj, 'memo', None)
        # getting the rank right - or at least trying to do so....
        rank = 1
        a=np.shape(np.shape(obj))[0] 
        if a == 4:
            rank = np.shape(obj)[3]
        self.rank = rank

    def __array_wrap__(self,obj,context=None):
        '''wrap it up'''
        b = np.ndarray.__array_wrap__(self, obj, context)
        #b.rank = self.rank * obj.rank
        rank = 1
        a=np.shape(np.shape(obj))[0] 
        if a == 4:
            rank = np.shape(obj)[3]
        b.rank = rank
        return b

    def integral(self):
        ''' Returns the integral of self '''
        return np.einsum('ijkl->',self)*self.grid.dV
        #return float(np.sum(self))*self.grid.dV


class DirectField(BaseField):
    spl_order = 3

    def __new__(cls, grid, memo="", rank=None, griddata_F=None, griddata_C=None, griddata_3d=None):
        if not isinstance(grid, DirectGrid):
            raise TypeError("the grid argument is not an instance of DirectGrid")
        obj = super().__new__(cls, grid, memo="", rank=rank, griddata_F=griddata_F, griddata_C=griddata_C, griddata_3d=griddata_3d)
        obj.spl_coeffs = None
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        #print("DirectScalarField __array_finalize__")
        #print(type(self))
        #print(type(obj))
        if obj is None: return
        super().__array_finalize__(obj)
        self.spl_coeffs = None

    def _calc_spline(self):
        padded_values = np.pad(self[:,:,:,0], ((self.spl_order,)), mode='wrap')
        self.spl_coeffs = ndimage.spline_filter(
            padded_values, order=self.spl_order)
        return

    def numerically_smooth_gradient(self):
        sq_self = np.sqrt(np.abs(self)) 
        grad = np.real(sq_self.standard_gradient())
        if grad.rank != np.shape(grad)[3]:
            raise ValueError("Gradient rank incompatible with shape")
        final = 2.0*sq_self*grad
        return final


    def super_smooth_gradient(self):
        reciprocal_self = self.fft()
        imag = (0 + 1j)
        nr = *self.grid.nr, 3
        grad_g = np.zeros(nr, dtype=complex)
        # Quantum Espresso way!
        for i in range(3):
            # FFT(\grad A) = i \vec(G) * FFT(A)
            grad_g[...,i] = reciprocal_self.grid.g[...,i] * (reciprocal_self[...,0]*imag) * np.exp(-reciprocal_self.grid.gg[...,0]*(0.1/2.0)**2  )
        grad_g = ReciprocalField(grid=self.grid.get_reciprocal(), rank=3, griddata_3d=grad_g)
        grad = np.real(grad_g.ifft(check_real=True))
        if grad.rank != np.shape(grad)[3]:
            raise ValueError("Standard Gradient: Gradient rank incompatible with shape")
        return grad


    def standard_gradient(self):
        reciprocal_self = self.fft()
        imag = (0 + 1j)
        nr = *self.grid.nr, 3
        grad_g = np.zeros(nr, dtype=complex)
        # Quantum Espresso way!
        for i in range(3):
            # FFT(\grad A) = i \vec(G) * FFT(A)
            grad_g[...,i] = reciprocal_self.grid.g[...,i] * (reciprocal_self[...,0]*imag)
        grad_g = ReciprocalField(grid=self.grid.get_reciprocal(), rank=3, griddata_3d=grad_g)
        grad = np.real(grad_g.ifft(check_real=True))
        if grad.rank != np.shape(grad)[3]:
            raise ValueError("Standard Gradient: Gradient rank incompatible with shape")
        return grad

    def divergence(self):
        if self.rank != 3:
            raise ValueError("Divergence: Rank incompatible")
        imag = (0 + 1j)
        nr = *self.grid.nr, 3
        grad_g = np.zeros(nr, dtype=complex)
        nr = *self.grid.nr, 1
        reciprocal_self = np.zeros(nr, dtype=complex)
        grad_g= self.grid.get_reciprocal().g*self.fft()*imag* np.exp(-self.grid.get_reciprocal().gg*(0.1/2.0)**2  )
        grad_g = ReciprocalField(grid=self.grid.get_reciprocal(), rank=3, griddata_3d=grad_g)
        grad = grad_g.ifft(check_real=True)
        div = np.sum(grad,axis=-1)
        div.rank = 1
        return DirectField(self.grid,rank=1,griddata_3d=div)

    def gradient(self,flag='smooth'):
        if self.rank > 1:
            raise Exception("gradient is only implemented for scalar fields")
        if flag is 'standard':
            return self.standard_gradient()
        elif flag is 'smooth':
            return self.numerically_smooth_gradient()
        elif flag is 'supersmooth':
            return self.super_smooth_gradient()



    def sigma(self):
        """
        \sigma(r) = |\grad rho(r)|^2
        """
        if self.rank > 1:
            raise Exception("sigma is only implemented for scalar fields")
        gradrho = self.gradient()
        griddata_3d = np.einsum("ijkl,ijkl->ijk", gradrho, gradrho)
        return DirectField(self.grid, rank=1, griddata_3d=np.abs(np.real(griddata_3d)))

    def fft(self):
        ''' Implements the Discrete Fourier Transform
        - Standard FFTs -
        Compute the N(=3)-dimensional discrete Fourier Transform
        Returns a new Grid_Function_Reciprocal
        '''
        dim = np.shape(np.shape(self))[0]
        #print("Shape = ",dim)
        if dim == 3:
            self.rank = 1
            nr = *self.grid.nr, self.rank
            other = self.reshape(nr)
            self = other
        else:
            nr = *self.grid.nr, self.rank
        dim = np.shape(np.shape(self))[0]
        #print("New Shape = ",dim)
        reciprocal_grid = self.grid.get_reciprocal()
        griddata_3d = np.zeros(nr, dtype=complex)
        for i in range(self.rank):
            #print("BLEAH! = ",np.shape(np.fft.fftn(self[:,:,:,i])))
            griddata_3d[:,:,:,i] = np.fft.fftn(self[:,:,:,i])*self.grid.dV
        return ReciprocalField(grid=reciprocal_grid, memo=self.memo, rank=self.rank, griddata_3d=griddata_3d)

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
            if self.span == 1:
                pad_tup = ((0,pad),(0,0),(0,0))
            elif self.span == 2:
                pad_tup = ((0,pad),(0,pad),(0,0))
            elif self.span == 3:
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
        if self.rank > 1:
            raise Exception("get_3dinterpolation is only implemented for scalar fields")

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
        new_lattice = self.grid.lattice #*LEN_CONV["Bohr"][self.grid.units]
        new_grid = DirectGrid(new_lattice, nr_new, units=self.grid.units)
        return DirectField(new_grid, self.memo, griddata_3d=new_values)

    def get_cut(self, r0, r1=None, r2=None, origin=None, center=None, nr=10):
        """
        general routine to get the arbitrary cuts of a Grid_Function_Base object in 1,2,
        or 3 dimensions. spline interpolation will be used.
            r0 = first vector (always required)
            r1 = second vector (required for 2D and 3D cuts)
            r2 = third vector (required for 3D cuts)
            origin = origin of the cut (don't specify center)
            center = center of the cut (don't specify origin)
            nr[i] = number points to discretize each direction ; i = 0,1,2
        r0, r1, r2, origin, center are instances of Coord
        """

        if self.rank > 1:
            raise Exception("get_cut is only implemented for scalar fields")

        span = 1

        do_center = False
        if origin is None and center is None:
            raise AttributeError("Specify either origin or center")
        elif origin is not None and center is not None:
            warnings.warn("Specified both origin and center, center will be ignored", DeprecationWarning)
        elif center is not None:
            do_center = True

        if do_center:
            x0 = center.to_crys()
        else:
            x0 = origin.to_crys()

        r0 = r0.to_crys()
        if do_center: x0 = x0 - 0.5*r0

        if r1 is not None:
            r1 = r1.to_crys()
            if do_center:
                x0 = x0 - 0.5*r1
            span += 1
            if r2 is not None:
                r2 = r2.to_crys()
                if do_center: x0 = x0 - 0.5*r2
                span += 1
        nrx = np.ones(3, dtype=int)
        if isinstance(nr, (int, float)):
            nrx[0:span] = nr
        elif isinstance(nr, (np.ndarray, list, tuple)):
            nrx[0:span] = np.asarray(nr, dtype=int)

        dr = np.zeros((3, 3), dtype=float)
        dr[0, :] = (r0) / nrx[0]
        if span > 1:
            dr[1, :] = (r1) / nrx[1]
            if span == 3:
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
        if span == 1:
            for i in range(3):
                if abs(v0[i]) > 1e-4:
                    j = i - 1
                    v1[j] = v0[i]
                    v1[i] = -v0[j]
                    v1 = v1 / np.sqrt(np.dot(v1,v1))
                    break
            v2 = np.cross(v0,v1)
            v2 = v2 / np.sqrt(np.dot(v2,v2))
        elif span  == 2:
            v1 = r1.to_cart()
            v2 = np.cross(v0,v1)
            v2 = v2/ np.sqrt(np.dot(v2,v2))
        elif span == 3:
            v1 = r1.to_cart()
            v2 = r2.to_cart()
        at[:,0] = v0
        at[:,1] = v1
        at[:,2] = v2

        cut_grid = DirectGrid(lattice=at, nr=nrx, origin=origin, units=x0.cell.units)

        if span == 1:
            values = values.reshape((a,))
        elif span == 2:
            values = values.reshape((a, b))
        elif span == 3:
            values = values.reshape((a, b, c))

        return DirectField(grid=cut_grid, memo=self.memo, griddata_3d=values)


class ReciprocalField(BaseField):

    def __new__(cls, grid, memo="", rank=None, griddata_F=None, griddata_C=None, griddata_3d=None):
        if not isinstance(grid, ReciprocalGrid):
            raise TypeError("the grid argument is not an instance of ReciprocalGrid")
        obj = super().__new__(cls, grid, memo="", rank=rank, griddata_F=griddata_F, griddata_C=griddata_C, griddata_3d=griddata_3d)
        obj.spl_coeffs = None
        return obj

    def __array_finalize__(self, obj):
        # Restore attributes when we are taking a slice
        if obj is None: return
        super().__array_finalize__(obj)
        self.spl_coeffs = None

    def ifft(self, check_real=False, force_real=False):
        '''
        Implements the Inverse Discrete Fourier Transform
        - Standard FFTs -
        Compute the N(=3)-dimensional inverse discrete Fourier Transform
        Returns a new Grid_Function
        '''
        direct_grid = self.grid.get_direct()
        nr = *self.grid.nr, self.rank
        griddata_3d = np.zeros(nr, dtype=complex)
        for i in range(self.rank):
            griddata_3d[...,i] = np.fft.ifftn(self[...,i])/direct_grid.dV
        if check_real:
            if np.isclose(np.imag(griddata_3d),0.,atol=1.e-16).all():
                griddata_3d = np.real(griddata_3d)
        if force_real:
            griddata_3d = np.real(griddata_3d)
        return DirectField(grid=direct_grid, memo=self.memo, rank=self.rank, griddata_3d=griddata_3d)




