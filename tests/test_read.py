import unittest
import numpy as np
from pbcpy.formats.qepp import PP
from pbcpy.optimization import Optimization
from pbcpy.functionals import FunctionalClass, TotalEnergyAndPotential
from pbcpy.constants import LEN_CONV
from pbcpy.formats.qepp import PP
from pbcpy.ewald import ewald
from pbcpy.grid import DirectGrid, ReciprocalGrid
from pbcpy.field import DirectField, ReciprocalField
from pbcpy.io.vasp import  read_POSCAR

def test_read():
    path_pp='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/OEPP/'
    path_rho='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/rho/'
    path_pos='tests/Benchmarks_TOTAL_ENERGY/GaAs_test/POSCAR/'
    pos1 = 'POSCAR_1'
    ions = read_POSCAR(path_pos+pos1, names=['Ga', 'As'])
    print(ions.pos)
    print(ions.nat)
    print(ions.labels)

if __name__ == "__main__":
    test_read()
        
        
