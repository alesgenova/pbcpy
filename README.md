# `pbcpy`
`pbcpy` is a Python3 package providing some useful tools when dealing with
molecules and materials under periodic boundary conditions (PBC).

Foundation of the package are the `Cell` and `Coord` classes, which define a unit cell under PBC, and a cartesian/crystal coordinate respectively.

`pbcpy` also provides tools to deal with quantities represented on an equally spaced grids, through the `Grid` and `Plot` classes. Operations such as interpolations or taking arbitrary 1D/2D/3D cuts are made very easy.

In addition, `pbcpy` exposes a fully periodic N-rank array, the `pbcarray`, which is derived from the `numpy.ndarray`.

Finally, `pbcpy` provides IO supportt to some common file formats:
- The Quantum Espresso `.pp` format (read only)
- The XCrySDen `.xsf` format (write only)

### `Cell` examples
A unit cell is defined by its lattice vectors. To create a `Cell` object we need to provide it a `3x3` matrix containing the lattice vectors (as columns).
```python
from pbcpy.base import Cell
at1 = np.identity(3)*10
cell1 = Cell(at=at1, origin=[0,0,0], units="Angstrom") # 10A cubic cell
# Valid units are "Angstrom", "Bohr", "nm", "m"
```

#### `Cell` attributes
```python
# the units
cell1.units
>>> 'Angstrom'

# the direct lattice
cell1.at
>>> array([[ 10.,   0.,   0.],
           [  0.,  10.,   0.],
           [  0.,   0.,  10.]])

# the reciprocal lattice
cell1.bg
>>> array([[ 0.1,  0. ,  0. ],
           [ 0. ,  0.1,  0. ],
           [ 0. ,  0. ,  0.1]])

# the volume
cell1.omega
>>> 1000.0
```

#### `Cell` methods
The `==` operator is implemented, you can compare two `Cell` objects even if they have different units
```python
at2 = np.identity(3)
cell2 = Cell(at=at2, origin=[0,0,0], units="nm") # 1nm cubic cell
cell2 == cell1
>>> True
```

### `Coord` examples
`Coord` is a `numpy.array` derived class, with some additional attributes and methods.
Coordinates in a periodic system are meaningless without the reference unit cell, this is why a `Coord` object also has an embedded `Cell` attribute.
Also, coordinates can be either `"Cartesian"` or `"Crystal"`.
```python
from pbcpy.base import Coord
pos1 = Coord(pos=[0.5,0.6,0.3], cell=cell1, ctype="Cartesian")
```

#### `Coord` attributes
```python
# the coordinate type (Cartesian or Crystal)
pos1.ctype
>>> 'Cartesian'

# the cell attribute is a Cell object
type(pos1.cell)
>>> pbcpy.base.Cell
```

#### `Coord` methods
`Coord` has several methods:
- `to_crys()`, `to_cart()` : convert to crystal or cartesian coordinates (returns new object).
- `conv(units)` : converts cartesian coordinates (and the associated cell) to new units (returns new object).
- `d_mic(other)` : Calculate the vector connecting two coordinates (from self to other), using the minimum image convention (MIC). The result is itself a coordinate.
- `dd_mic(other)` : Calculate the distance between two coordinates, using the MIC.
- `+`/`-` operators : Calculate the difference/sum between two coordinates without using the MIC. `units` and `ctype` conversions are automatically done as needed.
```python
pos1 = Coord(pos=[0.5,0.0,1.0], cell=cell1, ctype="Crystal")
pos2 = Coord(pos=[0.6,-1.0,3.0], cell=cell1, ctype="Crystal")

# convert to Crystal or Cartesian (returns new object)
pos1.to_cart() 
>>> Coord([  5.,   0.,  10.]) # the coordinate was already Cartesian, the result is still correct.
pos1.to_crys()
>>> Coord([ 0.5,  0. ,  1. ]) # the coordinate was already Crystal, the result is still correct.

## vector connecting two coordinates (using the minimum image convention), and distance
pos1.d_mic(pos2)
>>> Coord([ 0.1,  0. ,  0. ])
pos1.dd_mic(pos2)
>>> 1.0

## vector connecting two coordinates (without using the minimum image convention) and distance
pos2 - pos1
>>> Coord([ 0.1, -1. ,  2. ])
(pos2 - pos1).length()
>>> 22.383029285599392

```

### `Grid` class
`Grid` is a subclass of `Cell`, with additional attributes and methods to deal with quantities represented on a equally spaced grid.

```python
from pbcpy.grid import Grid
# A 10x10x10 Angstrom Grid, with 100x100x100 gridpoints
at1 = np.identity(3)*10
grid1 = Grid(at=at1, nr=[100,100,100], origin=[0,0,0], units="Angstrom"
```

#### `Grid` attributes
- All the attributes inherited from `Cell`
- `dV` : the volume of a single point, useful when calculating integral quantities
- `nr` : array, number of grid point for each direction
- `nnr` : total number of points in the grid
- `r` : cartesian coordinates at each grid point. A rank 3 array of type `Coord`
- `s` : crystal coordinates at each grid point. A rank 3 array of type `Coord`
```python
# The volume of each point
grid1.dV
>>> 0.001

# Grid points for each direction
grid1.nr
>>> [100, 100, 100]

# Total number of grid points
grid1.nnr
>>> 1000000

# Cartesian coordinates at each grid point
grid1.r
>>> Coord([[[[ 0. ,  0. ,  0. ],
         	 [ 0. ,  0. ,  0.1],
             [ 0. ,  0. ,  0.2],
             [ 0. ,  0. ,  0.3],
             ...]]])
grid1.r[0,49,10]
>>> Coord([ 0. ,  4.9,  1. ])

# Crystal coordinates at each grid point
grid1.s
>>> Coord([[[[ 0.  ,  0.  ,  0.  ],
       	     [ 0.  ,  0.  ,  0.01],
         	 [ 0.  ,  0.  ,  0.02],
             [ 0.  ,  0.  ,  0.03],
             ...]]])
```

### `Plot` class
The `Plot` class represents a scalar field on a `Grid`.

A `Plot` can be generated directly from Quantum Espresso postprocessing `.pp` files.
```python
# A Plot example
from pbcpy.grid import Plot
griddata = np.random.random(size=grid1.nr)
plot1 = Plot(grid=grid1, griddata_3d=griddata)

# Plots can be generated from Quantum Espresso files
from pbcpy.formats.qepp import PP
ppfile = "/home/alessandro/QE/FDE_Calc/dimer/H2O_0/density_ks.pp"
water = PP(filepp="/path/to/water.scf.pp").read()
plot2 = water.plot
```

#### `Plot` attributes
- `grid`, `ndim`
- `values` : 3D array containing the scalar field

#### `Plot` methods

- `get_3dinterpolation` : Interpolates the data to a different grid (returns a new `Plot` object). 3rd order spline interpolation.
- `get_plotcut` : Get 1D/2D/3D cuts of the data, by providing arbitraty vectors and an origin.

```python
# Interpolate the scalar field from one grid to another
plot2.values.shape
>>> (125, 125, 125)
plot3 = plot2.get_3dinterpolation([50,50,50])
plot3.values.shape
>>> (50, 50, 50)

# Get arbitrary cuts of the scalar field.
# In this example get the cut of the electron density in the plane of the water molecule
ppfile = "/home/alessandro/QE/FDE_Calc/dimer/H2O_0/density_ks.pp"
water = PP(ppfile).read()

O_pos = water.ions[0].pos
OH1_vec = water.ions[0].pos.d_mic(water.ions[1].pos)
OH2_vec = water.ions[0].pos.d_mic(water.ions[2].pos)

x0 = O_pos - 2*OH1_vec - 2*OH2_vec
r0 = x0 + 4*OH1_vec
r1 = x0 + 4*OH2_vec
# 1D

# 2D

# 3D
```
