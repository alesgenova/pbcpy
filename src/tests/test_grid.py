import unittest
import numpy as np
import sys
from env import PBC_SRC
#sys.path.insert(0,"../")
sys.path.insert(0,PBC_SRC)
from pbcpy.base import DirectCell, ReciprocalCell, Coord
from pbcpy.grid import DirectGrid, ReciprocalGrid
from pbcpy.constants import LEN_CONV

from common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell, make_triclinic_cell

class TestCell(unittest.TestCase):
    
    def test_orthorombic_cell(self):
        # run the same tests we ran on Cell onto Grid
        run_test_orthorombic(self, DirectGrid, nr=[10,10,10])
        # check if we can compare equality between Cell and Grid
        grid = make_orthorombic_cell(10,12,14,nr=[10,10,10],CellClass=DirectGrid)
        cell = make_orthorombic_cell(10,12,14,CellClass=DirectCell)
        self.assertEqual(grid,cell)


    def test_triclinic_cell(self):
        run_test_triclinic(self, DirectGrid, nr=[10,10,10])
        # check if we can compare equality between Cell and Grid
        grid = make_triclinic_cell(10,12,14,np.pi/2,np.pi/4,np.pi/3, nr=[10,10,10],CellClass=DirectGrid)
        cell = make_triclinic_cell(10,12,14,np.pi/2,np.pi/4,np.pi/3, CellClass=DirectCell)
        self.assertEqual(grid,cell)
