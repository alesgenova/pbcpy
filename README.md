# `pbcpy`
`pbcpy` is a Python3 package providing some useful tools when dealing with
molecules and materials under periodic boundary conditions (PBC).

Foundation of the package are the `Cell` and `Coord` classes, which define a unit cell under PBC, and a cartesian/crystal coordinate respectively.

`pbcpy` also provides tools to deal with quantities represented on an equally spaced grids, through the `Grid` and `Plot` classes. Operations such as interpolations or taking arbitrary 1D/2D/3D cuts are made very easy.

In addition, `pbcpy` exposes a fully periodic N-rank array, the `pbcarray`, which is derived from the `numpy.ndarray`.

Finally, `pbcpy` provides IO support to some common file formats:
- The Quantum Espresso `.pp` format (read only)
- The XCrySDen `.xsf` format (write only)

## `Cell` class
A unit cell is defined by its lattice vectors. To create a `Cell` object we need to provide it a `3x3` matrix containing the lattice vectors (as columns).

```python
>>> from pbcpy.base import Cell
>>> at1 = np.identity(3)*10
>>> cell1 = Cell(at=at1, origin=[0,0,0], units="Angstrom") # 10A cubic cell
# Valid units are "Angstrom", "Bohr", "nm", "m"
```

### `Cell` attributes
- `units` : the length units of the lattice vectors
- `at` : the lattice vectors (as columns)
- `bg` : the reciprocal vectors (as columns)
- `omega` : the volume of the cell
- 
```python
>>> cell1.units
'Angstrom'

# the direct lattice
>>> cell1.at
array([[ 10.,   0.,   0.],
       [  0.,  10.,   0.],
       [  0.,   0.,  10.]])

# the reciprocal lattice
>>> cell1.bg
array([[ 0.1,  0. ,  0. ],
       [ 0. ,  0.1,  0. ],
       [ 0. ,  0. ,  0.1]])

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
>>> grid1 = Grid(at=at1, nr=[100,100,100], origin=[0,0,0], units="Angstrom"
```

### `Grid` attributes
- All the attributes inherited from `Cell`
- `dV` : the volume of a single point, useful when calculating integral quantities
- `nr` : array, number of grid point for each direction
- `nnr` : total number of points in the grid
- `r` : cartesian coordinates at each grid point. A rank 3 array of type `Coord`
- `s` : crystal coordinates at each grid point. A rank 3 array of type `Coord`

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
>>> grid1.r[0,49,10]
Coord([ 0. ,  4.9,  1. ])

# Crystal coordinates at each grid point
>>> grid1.s
Coord([[[[ 0.  ,  0.  ,  0.  ],
  	     [ 0.  ,  0.  ,  0.01],
       	 [ 0.  ,  0.  ,  0.02],
         [ 0.  ,  0.  ,  0.03],
				           ...]]])
```

## `Plot` class
The `Plot` class represents a scalar field on a `Grid`.

A `Plot` can be generated directly from Quantum Espresso postprocessing `.pp` files.

```python
# A Plot example
>>> from pbcpy.grid import Plot
>>> griddata = np.random.random(size=grid1.nr)
>>> plot1 = Plot(grid=grid1, griddata_3d=griddata)

# Plots can be generated from Quantum Espresso files
>>> from pbcpy.formats.qepp import PP
>>> ppfile = "/home/alessandro/QE/FDE_Calc/dimer/H2O_0/density_ks.pp"
>>> water = PP(filepp="/path/to/water.scf.pp").read()
>>> plot2 = water.plot
```

### `Plot` attributes
- `grid`, `ndim`
- `values` : 3D array containing the scalar field

### `Plot` methods

- `get_3dinterpolation` : Interpolates the data to a different grid (returns a new `Plot` object). 3rd order spline interpolation.
- `get_plotcut(x0, r0, [r1], [r2], [nr])` : Get 1D/2D/3D cuts of the data, by providing arbitraty vectors and an origin.

```python
# Interpolate the scalar field from one grid to another
>>> plot2.values.shape
(125, 125, 125)

>>> plot3 = plot2.get_3dinterpolation([50,50,50])
>>> plot3.values.shape
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
# plot_cut is itself a Plot instance, and it can be either exported to an xsf file (see next session)
# or its values can be analized/manipulated in place.
>>> plot_cut.values.shape
(200, 200)
```

## `System` class
`System` is simply a class containing a `Cell` (or `Grid`), a set of atoms `ions`, and a `Plot`

### `System` attributes
- `name` : arbitrary name
- `ions` : collection of atoms and their coordinates
- `cell` : the unit cell of the system (`Cell` or `Grid`)
- `plot` : an optional `Plot` object.


## `pbcarray` class
`pbcarray` is a sublass of `numpy.ndarray`, and is suitable to represent periodic quantities, by including robust wrapping capabilities.
`pbcarray` can be of any rank, and it can be freely slices.

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
>>> ppfile = "/home/alessandro/QE/FDE_Calc/dimer/H2O_0/density_ks.pp"
>>> water = PP(ppfile).read() 
# the output of PP.read() is a System object.
```


### `XSF` class
`pbcpy` can write a `System` object into a XCrySDen  `.xsf` file.
```python
>>> xsffile = '/path/to/output.xsf'
>>> XSF(xsffile).write(system=water)

# ad additional plot parameter can be passed to XSF.write() in order to override the Plot in system.
# This is especially useful if one wants to output one system and an arbitrary cut of the grid,
# such as the one we generated before
>>> XSF(xsffile).write(system=water, plot=plot_cut)
```

