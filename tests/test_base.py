import unittest
import numpy as np

from pbcpy.base import DirectCell, ReciprocalCell, Coord, pbcarray
from pbcpy.constants import LEN_CONV

from .common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell

class TestCell(unittest.TestCase):
    
    def test_orthorombic_cell(self):
        print()
        print("*"*50)
        print("Testing orthorombic DirectCell")
        run_test_orthorombic(self, DirectCell)

    def test_triclinic_cell(self):
        print()
        print("*"*50)
        print("Testing triclinic DirectCell")
        run_test_triclinic(self, DirectCell)
    

class TestCoord(unittest.TestCase):

    def test_coord(self):
        print()
        print("*"*50)
        print("Testing Coord")
        # 9x12x18 cell
        ang2bohr = LEN_CONV["Angstrom"]["Bohr"]
        cell1 = make_orthorombic_cell(9,12,18,CellClass=DirectCell,units="Angstrom")
        rpos1 = np.array([3,6,12])*ang2bohr
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

        # test distance methods
        scoord1 = Coord(pos=[0.5,0.0,1.0], cell=cell1, basis="Crystal")
        scoord2 = Coord(pos=[0.6,-1.0,3.0], cell=cell1, basis="Crystal")
        dcoord = scoord1.d_mic(scoord2)
        # difference vector and distance using mic
        self.assertTrue(np.isclose(np.asarray(dcoord), [0.1,0.0,0.0]).all())
        self.assertAlmostEqual(scoord1.dd_mic(scoord2), 0.9*ang2bohr)
        # difference vector and distance without mic
        self.assertTrue(np.isclose(scoord2-scoord1, [0.1,-1.0,2.0]).all())
        self.assertAlmostEqual((scoord2-scoord1).length(), np.sqrt((0.1*9)**2+ (-1.*12)**2+(2.*18)**2)*ang2bohr)
        # start from a coord, add any lattice vector, calculate the difference (with MIC) from the start
        # it should be 0
        lcoord1 = Coord(pos=[1.,-5.,3.], cell=cell1, basis="Crystal")
        self.assertAlmostEqual(rcoord1.dd_mic(rcoord1+lcoord1), 0.0)
        # scalar multiply
        # wrong multiply
        with self.assertRaises(TypeError):
            rcoord1*lcoord1
        #wrong add
        cell2 = make_orthorombic_cell(9.3,12,18,CellClass=DirectCell,units="Angstrom")
        rcoord3 = Coord(pos=rpos1, cell=cell2, basis="Cartesian")
        with self.assertRaises(Exception):
            rcoord1+rcoord3


class TestPbcArray(unittest.TestCase):

    def test_pbcarray(self):
        print()
        print("*"*50)
        print("Testing PbcArray")
        dim = 50
        A = np.random.random((dim,dim))
        B = pbcarray(A)
        n = 5
        for _i in range(n):
            i = _i - n//2
            for _j in range(n):
                j = _j - n//2
                self.assertTrue(np.isclose(
                    A,
                    B[i*dim:i*dim+dim,j*dim:j*dim+dim]
                ).all())



if __name__ == "__main__":
    unittest.main()
