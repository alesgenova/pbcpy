import numpy as np
from ..grid import Grid, Plot
from ..system import System
from ..atom import Atom
from ..system import System


class PP(object):

    def __init__(self, filepp):
        self.filepp = filepp
        self.title = ''
        self.cutoffvars = {}
        # self.readpp()

    def read(self):

        with open(self.filepp) as filepp:
            # title
            self.title = filepp.readline()

            # nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp
            nrx = np.zeros(3, dtype=int)
            nr = np.zeros(3, dtype=int)
            nrx[0], nrx[1], nrx[2], nr[0], nr[1], nr[2], nat, ntyp = (
                int(x) for x in filepp.readline().split())

            # ibrav, celldm
            celldm = np.zeros(6, dtype=float)
            linesplt = filepp.readline().split()
            ibrav = int(linesplt[0])
            celldm = np.asarray(linesplt[1:], dtype=float)

            # at(:,i) three times
            if ibrav == 0:
                at = np.zeros((3, 3), dtype=float)
                for ilat in range(3):
                    linesplt = filepp.readline().split()
                    at[:, ilat] = np.asarray(linesplt, dtype=float)
                    # at[:,i] = (float(x) for x in filepp.readline().split())
                at *= celldm[0]
            else:
                at = self.celldm2at(ibrav, celldm)
            grid = Grid(at, nrx, units='Bohr')

            # gcutm, dual, ecut, plot_num
            # gcutm, dual, ecut, plot_num = (float(x) for x in filepp.readline().split())
            # plot_num = int(plot_num)
            filepp.readline()
            gcutm, dual, ecut, plot_num = 1., 1., 1., 0
            self.cutoffvars['ibrav'] = ibrav
            self.cutoffvars['celldm'] = celldm
            self.cutoffvars['gcutm'] = gcutm
            self.cutoffvars['dual'] = dual
            self.cutoffvars['ecut'] = ecut

            # ntyp
            atm = []
            zv = np.empty(ntyp, dtype=float)
            for ity in range(ntyp):
                linesplt = filepp.readline().split()
                atm.append(linesplt[1])
                zv[ity] = float(linesplt[2])
            # tau
            # tau = np.zeros((nat,3), dtype=float)
            # tau = np.zeros(3, dtype=float)
            # ityp = np.zeros(nat, dtype=int)
            atoms = []
            for iat in range(nat):
                linesplt = filepp.readline().split()
                # tau[iat,:] = np.asarray(linesplt[1:4],dtype=float)
                # ityp[iat] = int(linesplt[4]) -1
                tau = np.asarray(linesplt[1:4], dtype=float)
                ity = int(linesplt[4]) - 1
                atoms.append(Atom(Zval=zv[ity], label=atm[
                             ity], pos=tau * celldm[0], cell=grid))

            # self.atoms = Ions( nat, ntyp, atm, zv, tau*celldm[0], ityp, self.grid)

            # plot
            igrid = 0
            nnr = nrx[0] * nrx[1] * nrx[2]
            ppgrid = np.zeros(nnr, dtype=float)
            for line in filepp:
                line = line.split()
                npts = len(line)
                ppgrid[igrid:igrid + npts] = np.asarray(line, dtype=float)
                igrid += npts

            plot = Plot(grid, plot_num, griddata_pp=ppgrid)

            return System(atoms, grid, name=self.title, plot=plot)

    def writepp(self, system):

        with open(self.filepp, 'w') as filepp:
            val_per_line = 5

            # title
            # filepp.write(self.title)

            # nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp
            # mywrite(filepp, self.cell.nrx,False)
            # mywrite(filepp, self.cell.nr,False)
            # mywrite(filepp, [len(self.atoms.ions), len(self.atoms.species)],False)

            # ibrav, celldm
            # mywrite(filepp, self.cell.ibrav,True)
            # mywrite(filepp, self.cell.celldm,False)

            # at(:,i) three times
            # if self.cell.ibrav == 0 :
            #    for ilat in range(3):
            #        mywrite(filepp,self.cell.at[:,ilat],True)

            # gcutm, dual, ecut, plot_num
            # mywrite(filepp, self.cutoffvars['gcutm'],True)
            # mywrite(filepp, self.cutoffvars['dual'],False)
            # mywrite(filepp, self.cutoffvars['ecut'],False)
            # mywrite(filepp, self.plot.plot_num,False)

            # ntyp
            # for ity, spc in enumerate(self.atoms.species):
            #    mywrite(filepp,[ity+1,spc[0],spc[1]],True)

            # tau
            # for iat, ion in enumerate(self.atoms.ions):
            #    mywrite(filepp,iat+1,True)
            #    mywrite(filepp,ion.pos,False)
            #    mywrite(filepp,ion.typ+1,False)

            # plot
            nlines = self.grid.nnr // val_per_line
            grid_pp = self.plot.get_values_1darray(order='F')
            for iline in range(nlines):
                igrid = iline * val_per_line
                mywrite(filepp, grid_pp[igrid:igrid + val_per_line], True)
            igrid = (iline + 1) * val_per_line
            mywrite(filepp, grid_pp[igrid:self.grid.nnr], True)

    def celldm2at(self, ibrav, celldm):

        at = np.zeros((3, 3), dtype=float)

        if ibrav == 1:
            at = celldm[0] * np.identity(3)
        elif ibrav == 2:
            at[:, 0] = 0.5 * celldm[0] * np.array([-1., 0., 1.])
            at[:, 1] = 0.5 * celldm[0] * np.array([0., 1., 1.])
            at[:, 2] = 0.5 * celldm[0] * np.array([-1., 1., 0.])
        else:
            # implement all the other Bravais lattices
            pass

        return at


class Ions(object):

    def __init__(self, nat, ntyp, atm, zv, tau, ityp, cell):
        self.species = []
        self.ions = []
        for ity in range(ntyp):
            self.species.append([atm[ity], zv[ity]])

        for iat in range(nat):
            self.ions.append(Atom(Zval=zv[ityp[iat]], pos=tau[
                             :, iat], typ=ityp[iat], label=atm[ityp[iat]]))


def mywrite(fileobj, iterable, newline):
    if newline:
        fileobj.write('\n  ')
    # if len(iterable) > 1 :
    try:
        for ele in iterable:
            fileobj.write(str(ele) + '    ')
    except:
        fileobj.write(str(iterable) + '    ')
