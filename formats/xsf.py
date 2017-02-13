import numpy as np
from ..constants import LEN_CONV


class XSF(object):

    xsf_units = 'Angstrom'

    def __init__(self, filexsf):
        self.filexsf = filexsf
        self.title = ''
        self.cutoffvars = {}

    def write(self, system, plot=None):
        """
        Write a system object into an xsf file.
        Not all specifications of the xsf file format are implemented, they will
        be added as needed.
        So far it can:
            - write the system cell and atoms
            - write the 1D/2D/3D grid data
        """

        title = system.name
        cell = system.cell
        ions = system.ions

        # it can be useful to override the plot inside the system object,
        # for example if we want to plot a 2D/3D custom cut of the density grid
        if plot is None:
            plot = system.plot

        with open(self.filexsf, 'w') as fileout:
            self._write_header(fileout, title)
            self._write_cell(fileout, cell)
            self._write_coord(fileout, ions)
            self._write_datagrid(fileout, plot)


    def _write_header(self, fileout, title):
        mywrite(fileout, ("# ", title))
        mywrite(fileout, "CRYSTAL \n", True)

    def _write_cell(self, fileout, cell):
        mywrite(fileout, "PRIMVEC", True)
        for ilat in range(3):
            latt = cell.at[:, ilat] * LEN_CONV[cell.units][self.xsf_units]
            mywrite(fileout, latt, True)

    def _write_coord(self, fileout, ions):
        mywrite(fileout, "PRIMCOORD", True)
        mywrite(fileout, (len(ions), 1), True)
        for iat, atom in enumerate(ions):
            mywrite(fileout, (atom.label, atom.pos.conv(self.xsf_units)), True)

    def _write_datagrid(self, fileout, plot):
        ndim = plot.ndim # 2D or 3D grid?
        if ndim < 2:
            return # XSF format doesn't support one data grids
        val_per_line = 5
        values = plot.get_values_flatarray(pad=1, order='F')

        mywrite(fileout, "BEGIN_BLOCK_DATAGRID_{}D".format(ndim), True)
        mywrite(fileout, "{}d_datagrid_from_pbcpy".format(ndim), True)
        mywrite(fileout, "BEGIN_DATAGRID_{}D".format(ndim), True)
        nnr = len(values)
        origin = plot.grid.origin * LEN_CONV[plot.grid.units][self.xsf_units]
        if ndim == 3:
            mywrite(fileout, (plot.grid.nr[
                    0] + 1, plot.grid.nr[1] + 1, plot.grid.nr[2] + 1), True)
        elif ndim ==2:
            mywrite(fileout, (plot.grid.nr[
                    0] + 1, plot.grid.nr[1] + 1), True)
        mywrite(fileout, origin, True) # TODO, there might be an actual origin if we're dealing with a custom cut of the grid
        for ilat in range(ndim):
            latt = plot.grid.at[:, ilat] * LEN_CONV[plot.grid.units][self.xsf_units]
            mywrite(fileout, latt, True)

        nlines = nnr // val_per_line

        for iline in range(nlines):
            igrid = iline * val_per_line
            mywrite(fileout, values[igrid:igrid + val_per_line], True)
        igrid = (iline + 1) * val_per_line
        mywrite(fileout, values[igrid:nnr], True)

        mywrite(fileout, "END_DATAGRID_{}D".format(ndim), True)
        mywrite(fileout, "END_BLOCK_DATAGRID_{}D".format(ndim), True)


def mywrite(fileobj, iterable, newline=False):
    if newline:
        fileobj.write('\n  ')
    if isinstance(iterable, (np.ndarray, list, tuple)):
        for ele in iterable:
            mywrite(fileobj, ele)
            # fileobj.write(str(ele)+'    ')
    else:
        fileobj.write(str(iterable) + '    ')
