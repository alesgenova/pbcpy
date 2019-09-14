import numpy as np
from ..system import System
from ..atom import Atom
from ..base import BaseCell, DirectCell
from ..constants import LEN_CONV

BOHR2ANG   = LEN_CONV['Bohr']['Angstrom']

def read_POSCAR(infile, names=None):
    with open(infile) as fr:
        title = fr.readline()
        scale = list(map(float, fr.readline().split()))
        if len(scale) == 1 :
            scale = np.ones(3) * scale
        elif len(scale) == 3 :
            scale = np.asarray(scale)
        lat = []
        for i in range(2, 5):
            lat.append(list(map(float, fr.readline().split())))
        lat = np.asarray(lat).T/BOHR2ANG
        for i in range(3):
            lat[i] *= scale[i]
        lineL = fr.readline().split()
        if lineL[0].isdigit():
            typ = list(map(int, lineL))
        else:
            names = lineL
            typ = list(map(int, fr.readline().split()))
        if names is None :
            raise AttributeError("Must input the atoms names")
        Format = fr.readline().strip()[0]
        if Format == 'D' or Format == 'd' :
            Format = 'Crystal'
        elif Format == 'F' or Format == 'f' :
            Format = 'Cartesian'
        nat=sum(typ)
        pos = []
        i = 0
        for line in fr :
            i += 1
            if i > nat :
                break
            else :
                pos.append(list(map(float, line.split()[:3])))
    labels =[]
    for i in range(len(names)):
        labels.extend([names[i]] * typ[i])

    cell = DirectCell(lat)
    atoms = Atom(label=labels,pos=pos, cell=cell, basis = Format)

    return atoms
