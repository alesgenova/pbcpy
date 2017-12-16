import unittest
import numpy as np
import sys
from env import PBC_SRC
#sys.path.insert(0,"../")
sys.path.insert(0,PBC_SRC)
from pbcpy.base import DirectCell, ReciprocalCell, Coord
from pbcpy.constants import LEN_CONV

from common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell

class TestCell(unittest.TestCase):
    
    def test_orthorombic_cell(self):
        run_test_orthorombic(self, DirectCell)

    def test_triclinic_cell(self):
        run_test_triclinic(self, DirectCell)
    

class TestCoord(unittest.TestCase):

    def test_coord(self):
        # 9x12x18 cell
        cell1 = make_orthorombic_cell(9,12,18,CellClass=DirectCell,units="Angstrom")
        rpos1 = [3,6,12]
        #print(cell1.lattice)
        rcoord1 = Coord(pos=rpos1, cell=cell1, basis="Cartesian")
        #print(rcoord1)
        self.assertTrue(np.isclose(rcoord1, rcoord1.to_cart(),rtol=1.e-5).all())
        self.assertTrue(np.isclose(rcoord1, rcoord1.to_basis("R"),rtol=1.e-5).all())
        scoord1 = rcoord1.to_crys()
        #print(scoord1)
        self.assertTrue(np.isclose(np.asarray(scoord1), np.array([3/9,6/12,12/18]),rtol=1.e-5).all())
        self.assertTrue(np.isclose(scoord1, scoord1.to_crys(),rtol=1.e-5).all())
        self.assertTrue(np.isclose(scoord1, scoord1.to_basis("S"),rtol=1.e-5).all())
        rcoord2 = scoord1.to_cart()
        #print(rcoord2)
        self.assertTrue(np.isclose(rcoord2, rcoord1,rtol=1.e-5).all())


if __name__ == "__main__":
    unittest.main()
