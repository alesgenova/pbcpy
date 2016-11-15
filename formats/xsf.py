'''
    pbcpy is a python package to seamlessly tackle periodic boundary conditions.

    Copyright (C) 2016 Alessandro Genova (ales.genova@gmail.com).

    pbcpy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    pbcpy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with pbcpy.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
from ..constants import LEN_CONV


class XSF(object):

    def __init__(self, filexsf):
        self.filexsf = filexsf
        self.title = ''
        self.cutoffvars = {}

    def write(self, system):

        xsf_units = 'Angstrom'

        title = system.name
        cell = system.cell
        ions = system.ions
        plot = system.plot

        with open(self.filexsf, 'w') as fileout:

            mywrite(fileout, ("# ", title))

            mywrite(fileout, "CRYSTAL \n", True)

            mywrite(fileout, "PRIMVEC", True)
            for ilat in range(3):
                latt = cell.at[:, ilat] * LEN_CONV[cell.units][xsf_units]
                mywrite(fileout, latt, True)

            mywrite(fileout, "PRIMCOORD", True)
            mywrite(fileout, (len(ions), 1), True)
            for iat, atom in enumerate(ions):
                mywrite(fileout, (atom.label, atom.pos.conv(xsf_units)), True)

            mywrite(fileout, "BEGIN_BLOCK_DATAGRID_3D", True)
            val_per_line = 5
            values = plot.get_values_1darray(pad=1, order='F')

            mywrite(fileout, "3d_datagrid", True)
            mywrite(fileout, "BEGIN_DATAGRID_3D", True)
            nnr = len(values)
            mywrite(fileout, (plot.grid.nr[
                    0] + 1, plot.grid.nr[1] + 1, plot.grid.nr[2] + 1), True)
            mywrite(fileout, (0., 0., 0.), True)
            for ilat in range(3):
                latt = cell.at[:, ilat] * LEN_CONV[cell.units][xsf_units]
                mywrite(fileout, latt, True)

            nlines = nnr // val_per_line
            print(nnr)
            for iline in range(nlines):
                igrid = iline * val_per_line
                mywrite(fileout, values[igrid:igrid + val_per_line], True)
            igrid = (iline + 1) * val_per_line
            mywrite(fileout, values[igrid:nnr], True)

            mywrite(fileout, "END_DATAGRID_3D", True)
            mywrite(fileout, "END_BLOCK_DATAGRID_3D", True)


def mywrite(fileobj, iterable, newline=False):
    if newline:
        fileobj.write('\n  ')
    if isinstance(iterable, (np.ndarray, list, tuple)):
        for ele in iterable:
            mywrite(fileobj, ele)
            # fileobj.write(str(ele)+'    ')
    else:
        fileobj.write(str(iterable) + '    ')
