# `pbcpy`
`pbcpy` is a Python3 package providing some useful tools when dealing with
molecules and materials under periodic boundary conditions (PBC).

In addition, `pbcpy` exposes a fully periodic N-rank array, the `pbcarray`, which is derived from the `numpy.ndarray`.

Finally, `pbcpy` provides IO support to some common file formats:
- Quantum Espresso `.pp` format (read only)
- XCrySDen `.xsf` format (write only)

`pbcpy` has been developed @ [Pavanello Research Group](http://michelepavanello.com/) by:
- Alessandro Genova 
- Tommaso Pavanello

## Foundation of the package
- `Cell` and `Coord` classes which define a unit cell under PBC, and a cartesian/crystal coordinate respectively;
- `Grid` class defines a grid in a unit cell;
- `Grid_Space` class is more general than Grid class and defines both a grid in a unit cell and its reciprocal;
- `Grid_Function_Base` class defines functions on a grid;
- `Grid_Function` and `Grid_Function_Reciprocal` classes define functions on a grid space. Their domain is the "real" grid and the "reciprocal" grid, respectively.

## Installation

```
pip install pbcpy
```

## `Cell` class
A unit cell is defined by its lattice vectors. To create a `Cell` object we need to provide it a `3x3` matrix containing the lattice vectors (as columns).

```python
>>> from pbcpy.base import Cell
>>> at1 = np.identity(3)*10 # Make sure that at1 is of type numpy array.
>>> cell1 = Cell(at=at1, origin=[0,0,0], units="Angstrom") # 10A cubic cell
# Valid units are "Angstrom", "Bohr", "nm", "m"
```

### `Cell` attributes
- `units` : the length units of the lattice vectors
- `at` : the lattice vectors (as columns)
- `bg` : the inverse of matrix at
- `omega` : the volume of the cell
- `origin` : the origin of the Cartesian reference frame

```python
>>> cell1.units
'Angstrom'

# the lattice
>>> cell1.at
array([[ 10.,   0.,   0.],
       [  0.,  10.,   0.],
       [  0.,   0.,  10.]])

# the volume
>>> cell1.omega
1000.0
```

### `Cell` methods
- `==` operator : compare two `Cell` objects even if they have different units

```python
>>> at2 = np.identity(3)
>>> cell2 = Cell(at=at2, origin=[0,0,0], units="nm") # 1nm cubic cell
>>> cell2 == cell1
True
```

- `reciprocal_cell`: returns a new cell object that is the "reciprocal" cell of self

```python
>>> reciprocal_cell1 = cell1.reciprocal_cell()
>>> reciprocal_cell2 = cell2.reciprocal_cell()
>>> print(reciprocal_cell1.at)
array([[ 0.1,  0. ,  0. ],
       [ 0. ,  0.1,  0. ],
       [ 0. ,  0. ,  0.1]])

>>> print(reciprocal_cell2.at)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
```

## `Coord` class
`Coord` is a `numpy.array` derived class, with some additional attributes and methods.
Coordinates in a periodic system are meaningless without the reference unit cell, this is why a `Coord` object also has an embedded `Cell` attribute.
Also, coordinates can be either `"Cartesian"` or `"Crystal"`.

```python
>>> from pbcpy.base import Coord
>>> pos1 = Coord(pos=[0.5,0.6,0.3], cell=cell1, ctype="Cartesian")
```

### `Coord` attributes
- `ctype` : the coordinate type: `'Cartesian'` or `'Crystal'`.
- `cell` : the `Cell` object associated to the coordinates.

```python
# the coordinate type (Cartesian or Crystal)
>>> pos1.ctype
'Cartesian'

# the cell attribute is a Cell object
>>> type(pos1.cell)
pbcpy.base.Cell
```

### `Coord` methods
- `to_crys()`, `to_cart()` : convert to crystal or cartesian coordinates (returns new object).
- `conv(units)` : converts cartesian coordinates (and the associated cell) to new units (returns new object).
- `d_mic(other)` : Calculate the vector connecting two coordinates (from self to other), using the minimum image convention (MIC). The result is itself a coordinate.
- `dd_mic(other)` : Calculate the distance between two coordinates, using the MIC.
- `+`/`-` operators : Calculate the difference/sum between two coordinates without using the MIC. `units` and `ctype` conversions are automatically done as needed.

```python
>>> pos1 = Coord(pos=[0.5,0.0,1.0], cell=cell1, ctype="Crystal")
>>> pos2 = Coord(pos=[0.6,-1.0,3.0], cell=cell1, ctype="Crystal")

# convert to Crystal or Cartesian (returns new object)
>>> pos1.to_cart() 
Coord([  5.,   0.,  10.]) # the coordinate was already Cartesian, the result is still correct.
>>> pos1.to_crys()
Coord([ 0.5,  0. ,  1. ]) # the coordinate was already Crystal, the result is still correct.

## vector connecting two coordinates (using the minimum image convention), and distance
>>> pos1.d_mic(pos2)
Coord([ 0.1,  0. ,  0. ])
>>> pos1.dd_mic(pos2)
1.0

## vector connecting two coordinates (without using the minimum image convention) and distance
>>> pos2 - pos1
Coord([ 0.1, -1. ,  2. ])
>>> (pos2 - pos1).length()
22.383029285599392
```

## `Grid` class
`Grid` is a subclass of `Cell`, with additional attributes and methods to deal with quantities represented on a equally spaced grid.

```python
>>> from pbcpy.grid import Grid
# A 10x10x10 Angstrom Grid, with 100x100x100 gridpoints
>>> at1 = np.identity(3)*10
>>> grid1 = Grid(at=at1, nr=[100,100,100], origin=[0,0,0], units="Angstrom", convention='')
>>> grid2 = Grid(at=at1, nr=[100,100,100], origin=[0,0,0], units="Angstrom", convention='mic')
>>> grid3 = Grid(at=at1, nr=[100,100,100], origin=[0,0,0], units="Angstrom", convention='mic_scaled')
```

### `Grid` attributes
- All the attributes inherited from `Cell`
- `dV` : the volume of a single point, useful when calculating integral quantities
- `nr` : array, number of grid point for each direction
- `nnr` : total number of points in the grid
- `r` : cartesian coordinates at each grid point. A rank 3 array of type `Coord`
- `s` : crystal coordinates at each grid point. A rank 3 array of type `Coord`

NOTE:
It is possible to choose between 3 different conventions for coordinates:
- `mic` : 'mic'
    MIC convention.
- `mic_scaled` : 'mic_scaled'
    MIC convention. Each vector i is scaled by multiplying it for nr[i]
- `normal` : any other string would stick with this choice.
    NO MIC conversion.

```python
# The volume of each point
>>> grid1.dV
0.001

# Grid points for each direction
>>> grid1.nr
[100, 100, 100]

# Total number of grid points
>>> grid1.nnr
1000000

# Cartesian coordinates at each grid point
>>> grid1.r
Coord([[[[ 0. ,  0. ,  0. ],
       	 [ 0. ,  0. ,  0.1],
         [ 0. ,  0. ,  0.2],
         [ 0. ,  0. ,  0.3],
                        ...]]])
>>> grid2.r
Coord([[[[ 0. ,  0. ,  0. ],
         [ 0. ,  0. ,  0.1],
         [ 0. ,  0. ,  0.2],
         ..., 
         [ 0. ,  0. , -0.3],
         [ 0. ,  0. , -0.2],
         [ 0. ,  0. , -0.1]],
                        ...]]])

>>> grid3.r
Coord([[[[  0.,   0.,   0.],
         [  0.,   0.,  10.],
         [  0.,   0.,  20.],
         ..., 
         [  0.,   0., -30.],
         [  0.,   0., -20.],
         [  0.,   0., -10.]],
                        ...]]])

>>> grid1.r[0,49,99]
Coord([ 0. ,  4.9,  9.9])

>>> grid2.r[0,49,99]
Coord([ 0. ,  4.9, -0.1])

>>> grid3.r[0,49,99]
Coord([   0.,  490.,  -10.])

# Crystal coordinates at each grid point
>>> grid1.s
Coord([[[[ 0.  ,  0.  ,  0.  ],
  	     [ 0.  ,  0.  ,  0.01],
       	 [ 0.  ,  0.  ,  0.02],
         [ 0.  ,  0.  ,  0.03],
				           ...]]])
```

### `Grid` methods
- `reciprocal_grid`: Returns a new Grid object that is the "reciprocal" grid of self. The Cell is scaled properly to include the scaled (*self.nr) reciprocal grid points.
Note1: We need to use the 'physics' convention where bg^ T = 2 \pi * at^ {-1}
physics convention defines the reciprocal lattice to be
exp^ {i G \cdot R} = 1
Numpy uses the "crystallographer's" definition ('crystallograph')
which comes from defining the reciprocal lattice to be
e^ {2\pi i G \cdot R} =1
In this case bg^ T = at^ {-1}
We can use the 'physics' one with conv_type='physics' (*2pi)
and the right scale (*self.nr)
Note2: We have to use 'Bohr' units to avoid changing hbar value

- `crystal_coord_array(array)`: Returns a Coord in crystal coordinates

```python
>>> grid1.crystal_coord_array([0.5,0.5,0.5]).to_cart()
Coord([ 5.,  5.,  5.])
```

- `cartesian_coord_array(array)`: Returns a Coord in cartesian coordinates

```python
>>> grid1.cartesian_coord_array([0.5,0.5,0.5]).to_crys()
Coord([ 0.05,  0.05,  0.05])
```

- `square_dist_values(center_array)`: Returns a ndarray with
        square distance from center_array of
        grid points in cartesian coordinates
- `dist_values(center_array)`: Returns a ndarray with
        the distance from center_array of
        grid points in cartesian coordinates
- `gaussianValues(center_array,alpha)`: Returns a ndarray with
        the values of the gaussian
        (1/(alpha*sqrt(2pi)))*exp(-square_dist_values(center_array)/(2.0*alpha**2))
        centered on center_array

```python
>>> grid1.gaussianValues(center_array=[0.5,0.5,0.5],alpha=0.2)[10:52,50,50]
array([  2.76047418e-87,   5.36588917e-83,   8.12318018e-79,
         9.57716246e-75,   8.79374771e-71,   6.28836191e-67,
         3.50209107e-63,   1.51895085e-59,   5.13081536e-56,
         1.34975651e-52,   2.76535477e-49,   4.41237749e-46,
         5.48303280e-43,   5.30634407e-40,   3.99941388e-37,
         2.34759768e-34,   1.07319187e-31,   3.82082771e-29,
         1.05940963e-26,   2.28768780e-24,   3.84729931e-22,
         5.03896770e-20,   5.13988679e-18,   4.08311782e-16,
         2.52613554e-14,   1.21716027e-12,   4.56736020e-11,
         1.33477831e-09,   3.03794142e-08,   5.38488002e-07,
         7.43359757e-06,   7.99187055e-05,   6.69151129e-04,
         4.36341348e-03,   2.21592421e-02,   8.76415025e-02,
         2.69954833e-01,   6.47587978e-01,   1.20985362e+00,
         1.76032663e+00,   1.99471140e+00,   1.76032663e+00])
```

## `Grid_Space` class
The Grid_Space class defines both a grid in a unit cell and its reciprocal.

### `Grid_Space` attributes
- grid : Grid
        grid on direct space
- reciprocal_grid : Grid
        grid on reciprocal space
- nr : array of numbers used for discretization
- nnr : total number of subcells

### `Grid_Space` methods
- `clone` : Returns a new Grid_Space object, clone of self.

## `Grid_Function_Base` class
The `Grid_Function_Base` class represents a scalar field on a `Grid`.

Operations such as interpolations or taking arbitrary 1D/2D/3D cuts are made very easy.

A `Grid_Function_Base` can be generated directly from Quantum Espresso postprocessing `.pp` files.

```python
# A Grid_Function_Base example
>>> from pbcpy.grid_functions import Grid_Function_Base
>>> griddata = np.random.random(size=grid1.nr)
>>> base_func1 = Grid_Function_Base(grid=grid1, griddata_3d=griddata)

# Grid_Function_Bases can be generated from Quantum Espresso files
>>> from pbcpy.formats.qepp import PP
>>> water = PP(filepp="/path/to/density.pp").read()
>>> base_func2 = water.base_func
```

### `Grid_Function_Base` attributes
- `grid` : Represent the domain of the function
- `ndim` : The number of dimensions of the grid
- `values` : 3D array containing the scalar field

### `Grid_Function_Base` methods

- `get_3dinterpolation` : Interpolates the data to a different grid (returns a new `Grid_Function_Base` object). 3rd order spline interpolation.
- `get_base_funccut(x0, r0, [r1], [r2], [nr])` : Get 1D/2D/3D cuts of the data, by providing arbitraty vectors and an origin.

```python
# Interpolate the scalar field from one grid to another
>>> base_func1.values.shape
(100, 100, 100)

>>> base_func3 = base_func1.get_3dinterpolation([50,50,50])
>>> base_func3.values.shape
(50, 50, 50)

# Get arbitrary cuts of the scalar field.
# In this example get the cut of the electron density in the plane of the water molecule
>>> ppfile = "/path/to/density.pp"
>>> water = PP(ppfile).read()

>>> O_pos = water.ions[0].pos
>>> OH1_vec = water.ions[0].pos.d_mic(water.ions[1].pos)
>>> OH2_vec = water.ions[0].pos.d_mic(water.ions[2].pos)

>>> x0 = O_pos - 2*OH1_vec - 2*OH2_vec
>>> r0 = x0 + 4*OH1_vec
>>> r1 = x0 + 4*OH2_vec
>>> plot_cut = plot2.get_plotcut(x0=x0, r0=r0, r1=r1, nr=200)
# plot_cut is itself a Grid_Function_Base instance, and it can be either exported to an xsf file (see next session)
# or its values can be analized/manipulated in place.
>>> plot_cut.values.shape
(200, 200)
```

- `integral` : Returns the integral of self

```python
>>> base_func1.integral()
500.22870792181016
```

- `sumValues` : Returns a ndarray with self(x) + g(x)
- `dotValues` : Returns a ndarray with self(x)*g (g can be an array provided it has the same dimension of self)
- `expValues` : Returns a ndarray with exp(self(x))
- `exponentiationCnstValues` : Returns a ndarray with self(x)^ c
- `sumCnstValues` : Returns a ndarray with self(x)+c
- `linear_combinationValues` : Returns a ndarray with a*self(x)+b*g(x)

## `Grid_Function` class
The Grid_Function class represents a function on real space (the real grid in a grid space is the domain). Extends Grid_Function_Base (functions on generic grid).

### `Grid_Function` attributes
- `grid` : Grid
        Represent the domain of the function.
- `grid_space` : Grid_Space

### `Grid_Function` methods
- `clone` : Returns a new Grid_Function object, clone of self.
- `fft` : Returns a new Grid_Function_Reciprocal. Implements the Discrete Fourier Transform - Compute the N(=3)-dimensional discrete Fourier Transform. See the jupyter-notebook Real_and_Reciprocal_Spaces.ipynb for examples.
- `exp` : Implements exp(self(x))
        Returns a new Grid_Function
- `dot` : Implements self(x)*g(x) or self(x)*g number
        Returns a new Grid_Function
- `sum` : Implements self(x)+g(x)
        Returns a new Grid_Function
- `exponentiationCnst` : Implements self(x)^ c
        Returns a new Grid_Function
- `sumCnst` : Implements self(x)+c
        Returns a new Grid_Function
- `linear_combination` : Implements a*self(x)+b*g(x)
        Returns a new Grid_Function
- `dist` : Returns a new Grid_Function,
        the distance from p of the grid points
        in cartesian coordinates
- `sqr_dist` : Returns a new Grid_Function,
        the square distance from p of the grid points
        in cartesian coordinates
- `gaussian` : Returns a new Grid_Function,
        the gaussian (see grid.gaussianValues for the details)
        centered in center_array
- `real` : Returns a new Grid_Function
        with the real part of self(x)
- `imag` : Returns a new Grid_Function
        with the imaginary part of self(x)
- `divide_func` : Returns a new Grid_Function
        with self(x)/g(x), if g(x)==0 -> 0
- `invert` : Returns a new Grid_Function
        with g/self(x), if self(x)==0 -> 0
- `energy_density` : Returns a new Grid_Function
        real(ifft(fft(self(x)^ b)*kernel)*self(x)^ a*c)
- `energy_potential` : Returns a new Grid_Function
        real(ifft(fft(self(x)^ a)*kernel)*c)


## `Grid_Function_Reciprocal` class
The Grid_Function class represents a function on reciprocal space (the reciprocal grid in a grid space is the domain). Extends Grid_Function_Base (functions on generic grid).

### `Grid_Function_Reciprocal` attributes
- `grid` : Grid
        Represent the domain of the function.
- `grid_space` : Grid_Space

### `Grid_Function_Reciprocal` methods
- `clone` : Returns a new Grid_Function_Reciprocal object, clone of self.
- `ifft` : Returns a new Grid_Function. Implements the Inverse Discrete Fourier Transform - Compute the N(=3)-dimensional inverse discrete Fourier Transform. See the jupyter-notebook Real_and_Reciprocal_Spaces.ipynb for examples.
- `exp` : Implements exp(self(x))
        Returns a new Grid_Function_Reciprocal
- `dot` : Implements self(x)*g(x) or self(x)*g number
        Returns a new Grid_Function_Reciprocal
- `sum` : Implements self(x)+g(x)
        Returns a new Grid_Function_Reciprocal
- `exponentiationCnst` : Implements self(x)^ c
        Returns a new Grid_Function_Reciprocal
- `sumCnst` : Implements self(x)+c
        Returns a new Grid_Function_Reciprocal
- `linear_combination` : Implements a*self(x)+b*g(x)
        Returns a new Grid_Function_Reciprocal
- `dist` : Returns a new Grid_Function_Reciprocal,
        the distance from p of the grid points
        in cartesian coordinates
- `sqr_dist` : Returns a new Grid_Function_Reciprocal,
        the square distance from p of the grid points
        in cartesian coordinates
- `gaussian` : Returns a new Grid_Function_Reciprocal,
        the gaussian (see grid.gaussianValues for the details)
        centered in center_array
- `real` : Returns a new Grid_Function_Reciprocal
        with the real part of self(x)
- `imag` : Returns a new Grid_Function_Reciprocal
        with the imaginary part of self(x)
- `divide_func` : Returns a new Grid_Function_Reciprocal
        with self(x)/g(x), if g(x)==0 -> 0
- `invert` : Returns a new Grid_Function_Reciprocal
        with g/self(x), if self(x)==0 -> 0

## `System` class
`System` is simply a class containing a `Cell` (or `Grid`), a set of atoms `ions`, and a `Grid_Function_Base`

### `System` attributes
- `name` : arbitrary name
- `ions` : collection of atoms and their coordinates
- `cell` : the unit cell of the system (`Cell` or `Grid`)
- `plot` : an optional `Grid_Function_Base` object.


## `pbcarray` class
`pbcarray` is a sublass of `numpy.ndarray`, and is suitable to represent periodic quantities, by including robust wrapping capabilities.
`pbcarray` can be of any rank, and it can be freely sliced.

```python
# 1D example, but it is valid for any rank.
>>> from pbcpy.base import pbcarray
>>> import  matplotlib.pyplot as plt
>>> x = np.linspace(0,2*np.pi, endpoint=False, num=100)
>>> y = np.sin(x)
>>> y_pbc = pbcarray(y)
>>> y_pbc.shape
(100,) 							# y_pbc only has 100 elements, but we can freely do operations such as:
>>> plt.plot(y_pbc[-100:200])	# and get the expected result
```

## File Formats

### `PP` class
`pbcpy` can read Quantum Espresso post-processing `.pp` files.

```python
>>> water = PP(filepp='/path/to/density.pp').read() 
# the output of PP.read() is a System object.
```

### `XSF` class
`pbcpy` can write a `System` object into a XCrySDen  `.xsf` file.

```python
>>> XSF(filexsf='/path/to/output.xsf').write(system=water)

# an additional plot parameter can be passed to XSF.write() in order to override the Grid_Function_Base in system.
# This is especially useful if one wants to output one system and an arbitrary cut of the grid,
# such as the one we generated before
>>> XSF(filexsf='/path/to/output.xsf').write(system=water, plot=plot_cut)
```

