import numpy as np
from scipy import ndimage
from .base import Cell, Coord


class Grid(Cell):

    def __init__(self, at, nr, origin=np.array([0.,0.,0.]), units='Bohr'):
        super().__init__(at, origin, units)
        self.nr = np.asarray(nr)
        self.nnr = nr[0] * nr[1] * nr[2]
        self.dV = self.omega / self.nnr
        self.r = None
        self.s = None
        self._calc_gridpoints()

    def _calc_gridpoints(self):
        if self.r is None:

            s0 = np.linspace(0, 1, self.nr[0], endpoint=False)
            s1 = np.linspace(0, 1, self.nr[1], endpoint=False)
            s2 = np.linspace(0, 1, self.nr[2], endpoint=False)
            S = np.ndarray(shape=(self.nr[0], self.nr[
                           1], self.nr[2], 3), dtype=float)
            S[:, :, :, 0], S[:, :, :, 1], S[
                :, :, :, 2] = np.meshgrid(s0, s1, s2, indexing='ij')
            self.s = Coord(S, cell=self, ctype='Crystal')
            self.r = self.s.to_cart()

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


class Plot(object):
    # order of the spline interpolation
    spl_order = 3

    def __init__(self, grid, plot_num=0, griddata_pp=None, griddata_3d=None):
        self.grid = grid
        self.ndim = (grid.nr > 1).sum()
        self.plot_num = plot_num
        self.spl_coeffs = None
        if griddata_pp is None and griddata_3d is None:
            pass
        elif griddata_pp is not None:
            self.values = np.reshape(griddata_pp, grid.nr, order='F')
        elif griddata_3d is not None:
            self.values = griddata_3d

    def _calc_spline(self):
        padded_values = np.pad(self.values, ((self.spl_order,)), mode='wrap')
        self.spl_coeffs = ndimage.spline_filter(
            padded_values, order=self.spl_order)
        return

    def get_3dinterpolation(self, nr_new):
        """
        Interpolates the values of the plot on a cell with a different number
        of points, and returns a new plot object.
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
        return Plot(new_grid, self.plot_num, griddata_3d=new_values)

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
        general routine to get the arbitrary cuts of a Plot object in 1,2,
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

        return Plot(grid=cut_grid, plot_num=self.plot_num, griddata_3d=values)
