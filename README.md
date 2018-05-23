# PbcPy

[![PyPI version](https://img.shields.io/pypi/v/pbcpy.svg)](https://pypi.python.org/pypi/pbcpy/)
[![PyPI status](https://img.shields.io/pypi/status/pbcpy.svg)](https://pypi.python.org/pypi/pbcpy/)
[![pipeline status](https://gitlab.com/ales.genova/pbcpy/badges/master/pipeline.svg)](https://gitlab.com/ales.genova/pbcpy/pipelines)
[![coverage report](https://gitlab.com/ales.genova/pbcpy/badges/master/coverage.svg)](https://gitlab.com/ales.genova/pbcpy/pipelines)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`pbcpy` is a Python3 package providing some useful abstractions to deal with
molecules and materials under periodic boundary conditions (PBC).

In addition, `pbcpy` exposes a fully periodic N-rank array, the `pbcarray`, which is derived from the `numpy.ndarray`.

Finally, `pbcpy` provides IO support to some common file formats:
- Quantum Espresso `.pp` format (read only)
- XCrySDen `.xsf` format (write only)

## Index
- [Authors](#authors)
- [Fundamentals](#fundamentals)
- [Installation](#installation)
- [**DirectCell** and **ReciprocalCell** class](#directcell-and-reciprocalcell-class)
- [**Coord** class](#coord-class)
- [**DirectGrid** and **ReciprocalGrid** class](#directgrid-and-reciprocalgrid-class)
- [**DirectField** and **ReciprocalField** class](#directfield-and-reciprocalfield-class)
- [**System** class](#system-class)
- [**pbcarray** class](#pbcarray-class)
- [File Formats](#file-formats)


## Authors

`pbcpy` has been developed @ [Pavanello Research Group](http://michelepavanello.com/) by:
- **Alessandro Genova**

with contributions from:
- Tommaso Pavanello
- Michele Pavanello

## Fundamentals
- `DirectCell` and `Coord` classes which define a unit cell under PBC in real space, and a cartesian/crystal coordinate respectively;
- `ReciprocalCell` class which defines a cell in reciprocal space;
- `DirectGrid` and `ReciprocalGrid` classes, which are derived from `DirectCell` and `ReciprocalCell` and provide space discretization;
- `DirectField` and `ReciprocalField`, classes to represent a scalar (such as an electron density or a potential) and/or vector fields associated to either a `DirectGrid` or a `ReciprocalGrid`;

## Installation

Install `pbcpy` through `PyPI`
```
pip install pbcpy
```

Install the dev version from `gitlab`
```
git clone git@gitlab.com:ales.genova/pbcpy.git
```

NOTE: `pbcpy` is in the early stages of development, classes and APIs are bound to be changed without prior notice.

## `DirectCell` and `ReciprocalCell` class
A unit cell is defined by its lattice vectors. To create a `DirectCell` object we need to provide it a `3x3` matrix containing the lattice vectors (as columns).
`pbcpy` expects atomic units, a flexible units system might be addedd in the future.

```python
>>> from pbcpy.base import DirectCell, ReciprocalCell
>>> import numpy as np
>>> lattice = np.identity(3)*10 # Make sure that at1 is of type numpy array.
>>> cell1 = DirectCell(lattice=lattice, origin=[0,0,0]) # 10 Bohr cubic cell
```

### `DirectCell` and `ReciprocalCell` properties

- `lattice` : the lattice vectors (as columns)
- `volume` : the volume of the cell
- `origin` : the origin of the Cartesian reference frame

```python
# the lattice
>>> cell1.lattice
array([[ 10.,   0.,   0.],
       [  0.,  10.,   0.],
       [  0.,   0.,  10.]])

# the volume
>>> cell1.volume
1000.0
```

### `DirectCell` and `ReciprocalCell` methods
- `==` operator : compare two `Cell` objects

- `get_reciprocal`: returns a new `ReciprocalCell` object that is the "reciprocal" cell of self (if self is a `DirectCell`)
- `get_direct`: returns a new `DirectCell` object that is the "direct" cell of self (if self is a `ReciprocalCell`)

Note, by default the physics convention is used when converting between direct and reciprocal lattice:

```
\big[\text{reciprocal.lattice}\big]^T = 2\pi \cdot \big[\text{direct.lattice}\big]^{-1}
```

```python
>>> reciprocal_cell1 = cell1.get_reciprocal()
>>> print(reciprocal_cell1.lattice)
array([[ 0.62831853,  0. ,  0. ],
       [ 0. ,  0.62831853,  0. ],
       [ 0. ,  0. ,  0.62831853]])

>>> cell2 = reciprocal_cell1.get_direct()
>>> print(cell2.lattice)
array([[ 10.,  0.,  0.],
       [ 0.,  10.,  0.],
       [ 0.,  0.,  10.]])

>>> cell1 == cell2
True
```

## `Coord` class
`Coord` is a `numpy.array` derived class, with some additional attributes and methods.
Coordinates in a periodic system are meaningless without the reference unit cell, this is why a `Coord` object also has an embedded `DirectCell` attribute.
Also, coordinates can be either expressed in either a `"Cartesian"` or `"Crystal"` basis.

```python
>>> from pbcpy.base import Coord
>>> pos1 = Coord(pos=[0.5,0.6,0.3], cell=cell1, ctype="Cartesian")
```

### `Coord` attributes
- `basis` : the coordinate type: `'Cartesian'` or `'Crystal'`.
- `cell` : the `DirectCell` object associated to the coordinates.

```python
# the coordinate type (Cartesian or Crystal)
>>> pos1.basis
'Cartesian'

# the cell attribute is a Cell object
>>> type(pos1.cell)
pbcpy.base.DirectCell
```

### `Coord` methods
- `to_crys()`, `to_cart()` : convert `self` to crystal or cartesian basis (returns a new `Coord` object).
- `d_mic(other)` : Calculate the vector connecting two coordinates (from self to other), using the minimum image convention (MIC). The result is itself a `Coord` object.
- `dd_mic(other)` : Calculate the distance between two coordinates, using the MIC.
- `+`/`-` operators : Calculate the difference/sum between two coordinates without using the MIC. `basis` conversions are automatically performed when needed. The result is itself a `Coord` object.

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
0.99999999999999978

## vector connecting two coordinates (without using the minimum image convention) and distance
>>> pos2 - pos1
Coord([ 0.1, -1. ,  2. ])
>>> (pos2 - pos1).length()
22.383029285599392
```

## `DirectGrid` and `ReciprocalGrid` classes
`DirectGrid` and `ReciprocalGrid` are subclasses of `DirectGrid` and `ReciprocalGrid` respectively. `Grid`s inherit all the attributes and methods of their respective `Cell`s, and have a few of their own to deal with quantities represented on a equally spaced grid.

```python
>>> from pbcpy.grid import DirectGrid
# A 10x10x10 Bohr Grid, with 100x100x100 gridpoints
>>> lattice = np.identity(3)*10
>>> grid1 = DirectGrid(lattice=lattice, nr=[100,100,100], origin=[0,0,0])
```

### `Grid` attributes
- All the attributes inherited from `Cell`
- `dV` : the volume of a single point, useful when calculating integral quantities
- `nr` : array, number of grid point for each direction
- `nnr` : total number of points in the grid
- `r` : cartesian coordinates at each grid point. A rank 3 array of type `Coord` (`DirectGrid` only)
- `s` : crystal coordinates at each grid point. A rank 3 array of type `Coord` (`DirectGrid` only)
- `g` : G vector at each grid point (`ReciprocalGrid` only)
- `gg` : Square of G vector at each grid point (`ReciprocalGrid` only)

```python
# The volume of each point
>>> grid1.dV
0.001

# Grid points for each direction
>>> grid1.nr
array([100, 100, 100])

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

>>> grid1.r.shape
(100, 100, 100, 3)

>>> grid1.r[0,49,99]
Coord([ 0. ,  4.9,  9.9])

# Crystal coordinates at each grid point
>>> grid1.s
Coord([[[[ 0.  ,  0.  ,  0.  ],
  	 [ 0.  ,  0.  ,  0.01],
       	 [ 0.  ,  0.  ,  0.02],
         [ 0.  ,  0.  ,  0.03],
			  ...]]]])

# Since DirectGrid inherits from DirectCell, we can still use the get_reciprocal methos
reciprocal_grid1 = grid1.get_reciprocal()

# reciprocal_grid1 is an instance of ReciprocalGrid
>>> reciprocal_grid1.g
array([[[[ 0.  ,  0.  ,  0.  ],
         [ 0.  ,  0.  ,  0.01],
         [ 0.  ,  0.  ,  0.02],
         ..., 
         [ 0.  ,  0.  , -0.03],
         [ 0.  ,  0.  , -0.02],
         [ 0.  ,  0.  , -0.01]],
         		   ...]]])

>>> reciprocal_grid1.g.shape
(100, 100, 100, 3)

>>> reciprocal_grid1.gg
array([[[ 0.    ,  0.0001,  0.0004, ...,  0.0009,  0.0004,  0.0001],
        [ 0.0001,  0.0002,  0.0005, ...,  0.001 ,  0.0005,  0.0002],
        [ 0.0004,  0.0005,  0.0008, ...,  0.0013,  0.0008,  0.0005],
        ..., 
        [ 0.0009,  0.001 ,  0.0013, ...,  0.0018,  0.0013,  0.001 ],
        [ 0.0004,  0.0005,  0.0008, ...,  0.0013,  0.0008,  0.0005],
        [ 0.0001,  0.0002,  0.0005, ...,  0.001 ,  0.0005,  0.0002]],
        ...,
                                                                  ]])

>>> reciprocal_grid1.gg.shape
(100, 100, 100)                                          
```

## `DirectField` and `ReciprocalField` class
The `DirectField` and `ReciprocalField` classes represent a scalar field on a `DirectGrid` and `ReciprocalGrid` respectively. These classes are extensions of the `numpy.ndarray`.

Operations such as interpolations, fft and invfft, and taking arbitrary 1D/2D/3D cuts are made very easy.

A `DirectField` can be generated directly from Quantum Espresso postprocessing `.pp` files (see below).

```python
# A DirectField example
>>> from pbcpy.field import DirectField
>>> griddata = np.random.random(size=grid1.nr)
>>> field1 = DirectField(grid=grid1, griddata_3d=griddata)

# When importing a Quantum Espresso .pp files a DirectField object is created
>>> from pbcpy.formats.qepp import PP
>>> water_dimer = PP(filepp="/path/to/density.pp").read()
>>> rho = water_dimer.field
>>> type(rho)
pbcpy.field.DirectField
```

### `DirectField` attributes
- `grid` : Represent the grid associated to the field (it's a `DirectGrid` or `ReciprocalGrid` object)
- `span` : The number of dimensions of the grid for which the number of points is larger than 1
- `rank` : The number of dimensions of the quantity at each grid point
  - `1` : scalar field (e.g. the rank of rho is `1`)
  - `>1` : vector field (e.g. the rank of the gradient of rho is `3`)

```python
>>> type(rho.grid)
pbcpy.grid.DirectGrid

>>> rho.span
3

>>> rho.rank
1
# the density is a scalar field
```

### `DirectField` methods

- Any method inherited from `numpy.array`.
- `integral` : returns the integral of the field.
- `get_3dinterpolation` : Interpolates the data to a different grid (returns a new `DirectField` object). 3rd order spline interpolation.
- `get_cut(r0, [r1], [r2], [origin], [center], [nr])` : Get 1D/2D/3D cuts of the scalar field, by providing arbitraty vectors and an origin/center.
- `fft` : Calculates the Fourier transform of self, and returns an instance of `ReciprocalField`, which contains the appropriate `ReciprocalGrid`

```python
# Integrate the field over the whole grid
>>> rho.integral()
16.000000002898673 # the electron density of a water dimer has 16 valence electrons as expected

# Interpolate the scalar field from one grid to another
>>> rho.shape
(125, 125, 125)

>>> rho_interp = rho.get_3dinterpolation([90,90,90])
>>> rho_interp.shape
(90, 90, 90)

>> rho_interp.integral()
15.999915251442873


# Get arbitrary cuts of the scalar field.
# In this example get the cut of the electron density in the plane of the water molecule
>>> ppfile = "/path/to/density.pp"
>>> water_dimer = PP(ppfile).read()

>>> o_pos = water_dimer.ions[0].pos
>>> h1_pos = water_dimer.ions[1].pos
>>> h2_pos = water_dimer.ions[2].pos

>>> rho_cut = rho.get_cut(r0=o_h1_vec*4, r1=o_h2_vec*4, center=o_pos, nr=[100,100])

# plot_cut is itself a DirectField instance, and it can be either exported to an xsf file (see next session)
# or its values can be analized/manipulated in place.
>>> rho_cut.shape
(100,100)
>>> rho_cut.span
2
>>> rho_cut.grid.lattice
array([[ 1.57225214, -6.68207161, -0.43149218],
       [-1.75366585, -3.04623853,  0.8479004 ],
       [-7.02978121,  0.97509868, -0.30802502]])

# plot_cut is itself a Grid_Function_Base instance, and it can be either exported to an xsf file (see next session)
# or its values can be analized/manipulated in place.
>>> plot_cut.values.shape
(200, 200)

# Fourier transform of the DirectField
>>> rho_g = rho.fft()
>>> type(rho_g)
pbcpy.field.ReciprocalField
```

### `ReciprocalField` methods

- `ifft` : Calculates the inverse Fourier transform of self, and returns an instance of `DirectField`, which contains the appropriate `DirectGrid`

```python
# inv fft:
# recall that rho_g = fft(rho)
>>> rho1 = rho_g.ifft()
>>> type(rho1)
pbcpy.field.DirectField

>>> rho1.grid == rho.grid
True

>>> np.isclose(rho1, rho).all()
True
# as expected ifft(fft(rho)) = rho
```

## `System` class
`System` is simply a class containing a `DirectCell` (or `DirectGrid`), a set of atoms `ions`, and a `DirectField`

### `System` attributes
- `name` : arbitrary name
- `ions` : collection of atoms and their coordinates
- `cell` : the unit cell of the system (`DirectCell` or `DirectGrid`)
- `field` : an optional `DirectField` object.


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
`pbcpy` can read a Quantum Espresso post-processing `.pp` file into a `System` object.

```python
>>> water_dimer = PP(filepp='/path/to/density.pp').read() 
# the output of PP.read() is a System object.
```

### `XSF` class
`pbcpy` can write a `System` object into a XCrySDen  `.xsf` file.

```python
>>> XSF(filexsf='/path/to/output.xsf').write(system=water_dimer)

# an optional field parameter can be passed to XSF.write() in order to override the DirectField in system.
# This is especially useful if one wants to output one system and an arbitrary cut of the grid,
# such as the one we generated earlier
>>> XSF(filexsf='/path/to/output.xsf').write(system=water_dimer, field=rho_cut)
```

# Orbital-Free DFT capability
`pbcpy` has a built-in `Functional` class which can be populated with Kinetic Energy functionals (KEDFs), XC functionals, local Pseudo-Potentials and Hartree. LibXC is used for GGA functionals (except vW).
The LibXC interface relies on pylibxc, which needs to be installed. 
### install libxc
 - download / git clone libxc at 
 - make sure to have installed cmake and autoreconf

1) cd to libxc 
2) autoreconf -i
3) ./configure --prefix=/opt --libdir=/opt/lib --enable-shared --disable-fortran LIBS=-lm
4) make
5) sudo make install
6) sudo python setup.py install

### example
```python
# load needed classes / foundries
>>> from pbcpy.formats.qepp import PP 
>>> from pbcpy.semilocal_xc import PBE, LDA, XC, KEDF
>>> from pbcpy.local_pseudopotential import NuclearElectron
>>> from pbcpy.hartree import HartreeFunctional
# load electron density from file.pp and array of PP files (only Al atoms in this case)
>>> mol = PP(filepp='./file.pp').read()
>>> rho_of_r = mol.field
# obtain the local part of the PP
>>> NuclearElectron = NuclearElectron(mol.ions,rho_of_r,["./Al_lda.oe01.recpot","./Al_lda.oe01.recpot"])
# Hartree potential / energy
>>> Hartree = HartreeFunctional(rho_of_r)
# LDA XC
>>> ExchangeCorrelation_LDA = LDA(rho_of_r,polarization='unpolarized')
# PBE XC
>>> ExchangeCorrelation_PBE = PBE(rho_of_r,polarization='unpolarized')
# LC94 KEDF
>>> LC94 = KEDF(density=rho_of_r,k_str='gga_k_lc94',polarization='unpolarized')
# add the potential to obtain the total *effective* KS potential
>>> KS_effective = NuclearElectron.sum(Hartree).sum(PBE)
>>> potential = KS_effective.potential
>>> energydensity = KS_effective.energydensity
```
