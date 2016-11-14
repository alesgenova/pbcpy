import numpy as np
from scipy import ndimage
from pbc.base import Cell, Coord

class Grid(Cell):

    def __init__(self, at, nr, units='Bohr'):
        super().__init__(at, units)
        self.nr = nr
        self.nnr = nr[0]*nr[1]*nr[2]
        self.dV = self.omega / self.nnr
        self.r = None
        self.s = None
        self.calc_gridpoints()

    def calc_gridpoints(self):
        if self.r is None :

            s0 = np.linspace(0, 1, self.nr[0], endpoint = False)
            s1 = np.linspace(0, 1, self.nr[1], endpoint = False)
            s2 = np.linspace(0, 1, self.nr[2], endpoint = False)
            S = np.ndarray(shape=(self.nr[0],self.nr[1],self.nr[2],3),dtype=float)
            S[:,:,:,0], S[:,:,:,1], S[:,:,:,2] = np.meshgrid(s0,s1,s2,indexing='ij')
            self.s = Coord(S, cell = self, ctype = 'Crystal')
            self.r = self.s.to_cart()
            
    def calc_mask(self,ref_points):
        
        cutr = 1.1
        
        mask = np.ones(self.nr,dtype=float)
        for i in range(self.nr[0]):
            for j in range(self.nr[1]):
                for k in range(self.nr[2]):
                    for point in ref_points:
                        point = Coord(point,self)
                        #print(point)
                        dd = self.r[i,j,k].d_mic(point).lenght()
                        if dd < cutr :
                            
                            mask[i,j,k] = 0.
        return mask
            



class Plot(object):

    spl_order = 3

    def __init__(self, grid, plot_num, grid_pp=None, grid_3d=None ):
        self.grid = grid
        self.plot_num = plot_num
        self.spl_coeffs = None
        if grid_pp is None and grid_3d is None :
            pass
        elif grid_pp is not None :
            self.values = np.reshape(grid_pp,grid.nr,order='F')
        elif grid_3d is not None :
            self.values = grid_3d

    def calc_spline(self) :
        padded_values = np.pad(self.values,((self.spl_order,)),mode='wrap')
        self.spl_coeffs = ndimage.spline_filter(padded_values, order=self.spl_order)
        return


    def get_3dinterpolation(self, nr_new) :
        '''Interpolates the values of the plot on a cell with a different number of points, and returns a new plot object.'''
        if self.spl_coeffs is None :
            self.calc_spline()
        x = np.linspace(0,1,nr_new[0],endpoint=False)*self.grid.nr[0] + self.spl_order
        y = np.linspace(0,1,nr_new[1],endpoint=False)*self.grid.nr[1] + self.spl_order
        z = np.linspace(0,1,nr_new[2],endpoint=False)*self.grid.nr[2] + self.spl_order
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        new_values = ndimage.map_coordinates(self.spl_coeffs, [X, Y, Z],mode='wrap')
        new_grid = Grid(self.grid.at, nr_new, units=self.grid.units)
        return Plot(new_grid, self.plot_num, grid_3d=new_values)


    def get_value_at_points(self, points) :
        '''points is in crystal coordinates'''
        if self.spl_coeffs is None :
            self.calc_spline()
        for ipol in range(3):
            # restrict crystal coordinates to [0,1)
            points[:,ipol] = (points[:,ipol]%1)*self.grid.nr[ipol] + self.spl_order
        values = ndimage.map_coordinates(self.spl_coeffs, [points[:,0],points[:,1],points[:,2]],mode='wrap')
        return values


    def get_values_1darray(self, pad=0, order='F'):
        if pad > 0:
            vals = np.pad(self.values,((0,pad)),mode='wrap')
        else :
            vals = self.values
        nr = vals.shape
        nnr = nr[0]*nr[1]*nr[2]
        print(nr, nnr)
        return np.reshape(vals,nnr,order=order)


    def get_plotcut(self, x0, r0, r1=None, r2=None, nr=10):
        '''general routine to get the arbitrary cuts of a Plot object in 1,2 or 3 dimensions. spline interpolation will be used.
            x0 = origin of the cut
            r0 = first vector (always required)
            r1 = second vector (required for 2D and 3D cuts)
            r2 = third vector (required for 3D cuts)
            nr[i] = number points to discretize each direction ; i = 0,1,2
            x0, r0, r1, r2 are all in crystal coordinates'''

        ndim =1

        x0 = x0.to_crys()
        r0 = r0.to_crys()
        if r1 is not None:
            r1 = r1.to_crys()
            ndim += 1
            if r2 is not None:
                r2 = r2.to_crys()
                ndim += 1
        nrx = np.ones(3,dtype=int)
        if isinstance(nr,(int,float)):
            nrx[0:ndim] = nr
        elif isinstance(nr,(np.ndarray,list,tuple)):
            nrx[0:ndim] = np.asarray(nr,dtype=int)

        dr = np.zeros((3,3),dtype=float)
        dr[0,:] = (r0-x0)/nrx[0]
        if ndim > 1 :
            dr[1,:] = (r1-x0)/nrx[1]
            if ndim == 3 :
                dr[2,:] = (r2-x0)/nrx[2]
        points = np.zeros((nrx[0],nrx[1],nrx[2],3))
        axis = []
        for ipol in range(3):
            axis.append(np.zeros((nrx[ipol],3)))
            for ir in range(nrx[ipol]):
                axis[ipol][ir,:] = x0 + ir*dr[ipol]

        for i in range(nrx[0]):
            for j in range(nrx[1]):
                for k in range(nrx[2]):
                    points[i,j,k,:] = axis[0][i,:] + axis[1][j,:] + axis[2][k,:]

        a,b,c,d = points.shape
        points = points.reshape((a*b*c,3))

        values = self.get_value_at_points(points)

        if ndim ==1 :
            values = values.reshape((a))
        elif ndim == 2 :
            values = values.reshape((a,b))
        elif ndim == 3 :
            values = values.reshape((a,b,c))


        return values

