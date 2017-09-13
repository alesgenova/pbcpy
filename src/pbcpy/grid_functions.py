import numpy as np
from scipy import ndimage
from .grid import Grid, Grid_Space

class Grid_Function_Base(object):
    '''
    Object representing a function on a grid
    (Cell (lattice) plus discretization)

    Attributes
    ----------
    grid : Grid
        Represent the domain of the function

    ndim : number of dimensions of the grid

    plot_num :

    spl_coeffs :

    values : ndarray
        the values of the function

    '''
    # order of the spline interpolation
    spl_order = 3

    def __init__(self, grid, plot_num=0, griddata_pp=None, griddata_3d=None):
        self.grid = grid
        self.ndim = (grid.nr > 1).sum()
        self.plot_num = plot_num
        self.spl_coeffs = None
        if griddata_pp is None and griddata_3d is None:
            self.values = None
        elif griddata_pp is not None:
            self.values = np.reshape(griddata_pp, grid.nr, order='F')
        elif griddata_3d is not None:
            self.values = griddata_3d

    def integral(self):
        ''' Returns the integral of self '''
        return np.einsum('ijk->',self.values)*self.grid.dV

    def sumValues(self,g):
        ''' Returns an ndarray with f(x) + g(x) '''
        if isinstance(g, type(self)):
            return self.values+g.values
        else:
            return Exception

    def dotValues(self,g):
        ''' Returns an ndarray with f(x)*g (g can be a function) '''
        if isinstance(g, type(self)):
            return self.values*g.values
        elif isinstance(g, (int,float,complex)):
            return self.values*g
        else:
            return Exception

    def expValues(self):
        ''' Returns exp(f(x)) '''
        return np.exp(self.values)

    def exponentiationCnstValues(self,c=1):
        ''' Returns f(x)^c '''
        if isinstance(c, (int,float,complex)):
            return self.values**c
        else:
            return Exception

    def sumCnstValues(self,c=0):
        ''' Returns f(x)+c '''
        if isinstance(c, (int,float,complex)):
            return self.values+c
        else:
            return Exception

    def linear_combinationValues(self,g,a,b):
        ''' Returns a*f(x)+b*g(x) '''
        if isinstance(g, type(self)) and isinstance(a, (int,float,complex)) and isinstance(b, (int,float,complex)):
            return (a*self.values)+(b*g.values)
        else:
            return Exception

    def _calc_spline(self):
        padded_values = np.pad(self.values, ((self.spl_order,)), mode='wrap')
        self.spl_coeffs = ndimage.spline_filter(
            padded_values, order=self.spl_order)
        return

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
        new_grid = Grid(self.grid.at, nr_new, units=self.grid.units)
        return self.__class__(new_grid, self.plot_num, griddata_3d=new_values)

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
            vals = np.pad(self.values, (0,pad), mode='wrap')
        else:
            vals = self.values
        nr = vals.shape
        nnr = 1
        for n in nr:
            nnr *= n
        #nnr = nr[0] * nr[1] * nr[2]
        print(nr, nnr)
        return np.reshape(vals, nnr, order=order)

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

        cut_grid = Grid(at=at, nr=nrx, origin=origin, units=x0.cell.units)

        if ndim == 1:
            values = values.reshape((a))
        elif ndim == 2:
            values = values.reshape((a, b))
        elif ndim == 3:
            values = values.reshape((a, b, c))

        return self.__class__(grid=cut_grid, plot_num=self.plot_num, griddata_3d=values)


class Grid_Function_Reciprocal(Grid_Function_Base):
    '''
    Object representing a function on reciprocal space (the reciprocal grid in a grid space is the domain)

    extends Grid_Function_Base (functions on generic grid)

    Attributes
    ----------
    grid : Grid
        Represent the domain of the function

    grid_space : Grid_Space

    '''

    def __init__(self, grid_space, plot_num=0, griddata_pp=None, griddata_3d=None):
        self.grid_space = grid_space
        self.grid = grid_space.reciprocal_grid
        super().__init__(self.grid, plot_num, griddata_pp, griddata_3d)

    def clone(self):
        return Grid_Function_Reciprocal(self.grid_space,plot_num=self.plot_num, griddata_pp=self.griddata_pp, griddata_3d=self.griddata_3d)

    def ifft(self):
        '''
        Implements the Inverse Discrete Fourier Transform
        - Standard FFTs -
        Compute the N(=3)-dimensional inverse discrete Fourier Transform
        Returns a new Grid_Function
        '''
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=np.fft.ifftn(self.values)/self.grid_space.grid.dV)

    def exp(self):
        ''' Implements exp(f(x))
        Returns a new Grid_Function_Reciprocal '''
        values = self.expValues()
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def dot(self,g):
        '''Implements f(x)*g(x) or f(x)*g number
        Returns a new Grid_Function_Reciprocal'''
        values = self.dotValues(g)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def sum(self,g):
        '''Implements f(x)+g(x)
        Returns a new Grid_Function_Reciprocal'''
        values = self.sumValues(g)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def exponentiationCnst(self,c=1):
        '''Implements f(x)^c
        Returns a new Grid_Function_Reciprocal'''
        values = self.exponentiationCnstValues(c)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def sumCnst(self,c=0):
        '''Implements f(x)+c
        Returns a new Grid_Function_Reciprocal'''
        values = self.sumCnstValues(c)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def linear_combination(self,g,a,b):
        ''' Implements a*f(x)+b*g(x)
        Returns a new Grid_Function_Reciprocal'''
        values = self.linear_combinationValues(g,a,b)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def dist(self,p=[0.,0.,0.]):
        '''Returns a new Grid_Function_Reciprocal,
        the distance from p of the grid points
        in cartesian coordinates'''
        values = self.grid.dist_values(p)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def sqr_dist(self,p=[0.,0.,0.]):
        '''Returns a new Grid_Function_Reciprocal,
        the square distance from p of the grid points
        in cartesian coordinates'''
        values = self.grid.square_dist_values(p)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def real(self):
        '''Returns a new Grid_Function_Reciprocal
        with the real part of f(x)'''
        values = np.real(self.values)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def imag(self):
        '''Returns a new Grid_Function_Reciprocal
        with the imaginary part of f(x)'''
        values = np.imag(self.values)
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)

    def divide_func(self,g):
        '''Returns a new Grid_Function_Reciprocal
        with f(x)/g(x), if g(x)==0 -> 0'''
        if isinstance(g, Grid_Function_Reciprocal):
            values = np.divide(self.values, g.values, out=np.zeros_like(self.values), where=g.values!=0)
            return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)
        else:
            print('Bad input on divide method')
            print("g type = %s" % type(g))
            return Exception

    def invert(self,g=1.):
        '''Returns a new Grid_Function_Reciprocal
        with g/f(x), if f(x)==0 -> 0'''
        if isinstance(g, (int,float,complex)):
            values = np.divide(g, self.values, out=np.zeros_like(self.values), where=self.values!=0)
            return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=values)
        else:
            print('Bad input on invert method')
            print("g type = %s" % type(g))
            return Exception

class Grid_Function(Grid_Function_Base):
    '''
    Object representing a function on real space (the real grid in a grid space is the domain)

    extends Grid_Function_Base (functions on generic grid)

    Attributes
    ----------
    grid : Grid
        Represent the domain of the function

    grid_space : Grid_Space'''

    def __init__(self, grid_space, plot_num=0, griddata_pp=None, griddata_3d=None):
        self.grid_space = grid_space
        self.grid = grid_space.grid
        super().__init__(self.grid, plot_num, griddata_pp, griddata_3d)

    def clone(self):
        return Grid_Function(self.grid_space,plot_num=self.plot_num, griddata_pp=self.griddata_pp, griddata_3d=self.griddata_3d)

    def fft(self):
        ''' Implements the Discrete Fourier Transform
        - Standard FFTs -
        Compute the N(=3)-dimensional discrete Fourier Transform
        Returns a new Grid_Function_Reciprocal
        '''
        return Grid_Function_Reciprocal(self.grid_space, self.plot_num, griddata_3d=np.fft.fftn(self.values)*self.grid.dV)

    def exp(self):
        ''' Implements exp(f(x))
        Returns a new Grid_Function'''
        values = self.expValues()
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def dot(self,g):
        '''Implements f(x)*g(x) or f(x)*g number
        Returns a new Grid_Function'''
        values = self.dotValues(g)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def sum(self,g):
        '''Implements f(x)+g(x)
        Returns a new Grid_Function'''
        values = self.sumValues(g)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def exponentiationCnst(self,c=1):
        '''Implements f(x)^c
        Returns a new Grid_Function'''
        values = self.exponentiationCnstValues(c)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def sumCnst(self,c=0):
        '''Implements f(x)+c
        Returns a new Grid_Function'''
        values = self.sumCnstValues(c)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def linear_combination(self,g,a,b):
        ''' Implements a*f(x)+b*g(x)
        Returns a new Grid_Function'''
        values = self.linear_combinationValues(g,a,b)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def dist(self,p=[0.,0.,0.]):
        '''Returns a new Grid_Function,
        the distance from p of the grid points
        in cartesian coordinates'''
        values = self.grid.dist_values(p)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def sqr_dist(self,p=[0.,0.,0.]):
        '''Returns a new Grid_Function,
        the square distance from p of the grid points
        in cartesian coordinates'''
        values = self.grid.square_dist_values(p)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def gaussian(self,alpha=0.1,center_array=[0.,0.,0.]):
        '''Returns a new Grid_Function,
        the gaussian (see grid.gaussianValues for the details)
        centered in center_array'''
        values = self.grid.gaussianValues(alpha,center_array)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def real(self):
        '''Returns a new Grid_Function
        with the real part of f(x)'''
        values = np.real(self.values)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def imag(self):
        '''Returns a new Grid_Function
        with the imaginary part of f(x)'''
        values = np.imag(self.values)
        return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)

    def divide_func(self,g):
        '''Returns a new Grid_Function
        with f(x)/g(x), if g(x)==0 -> 0'''
        if isinstance(g, Grid_Function):
            values = np.divide(self.values, g.values, out=np.zeros_like(self.values), where=g.values!=0)
            return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)
        else:
            print('Bad input on divide_func method')
            print("g type = %s" % type(g))
            return Exception

    def invert(self,g=1.):
        '''Returns a new Grid_Function
        with g/f(x), if f(x)==0 -> 0'''
        if isinstance(g, (int,float,complex)):
            values = np.divide(g, self.values, out=np.zeros_like(self.values), where=self.values!=0)
            return Grid_Function(self.grid_space, self.plot_num, griddata_3d=values)
        else:
            print('Bad input on invert method')
            print("g type = %s" % type(g))
            return Exception

    def energy_density(self,kernel,a,b,c=1):
        '''Returns a new Grid_Function
        real(ifft(fft(f(x)^b)*kernel)*f(x)^a*c)'''

        if isinstance(kernel, (Grid_Function_Reciprocal,int,float,complex)) and isinstance(a, (int,float,complex)) and isinstance(b, (int,float,complex)) and isinstance(c, (int,float,complex)):
            first_exp = self.exponentiationCnst(a)
            second_exp = self.exponentiationCnst(b)
            second_exp_tranf = second_exp.fft()
            prod = second_exp_tranf.dot(kernel)
            inverse_transf = prod.ifft()
            return inverse_transf.dot(first_exp).dot(c).real()
        else:
            print('Bad input on energy_density method')
            print("kernel type = %s" % type(kernel))
            return Exception

    def energy_potential(self,kernel,a,c=1.):
        '''Returns a new Grid_Function
        real(ifft(fft(f(x)^a)*kernel)*c)'''

        if isinstance(kernel, (Grid_Function_Reciprocal,int,float,complex)) and isinstance(a, (int,float,complex)) and isinstance(c, (int,float,complex)):
            exp = self.exponentiationCnst(a)
            exp_tranf = exp.fft()
            prod = exp_tranf.dot(kernel)
            inverse_transf = prod.ifft()
            return inverse_transf.dot(c).real()
        else:
            print('Bad input on energy_potential method')
            print("kernel type = %s" % type(kernel))
            return Exception
