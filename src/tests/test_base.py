import unittest
import numpy as np
import sys
sys.path.insert(0,"../")
from pbcpy.base import DirectCell, ReciprocalCell
from pbcpy.constants import LEN_CONV

class TestDirectCell(unittest.TestCase):
    
    def test_init(self):
        # 10A cubic cell
        edge = 10
        cell1 = DirectCell(lattice=np.identity(3)*edge, origin=[0,0,0], units="Angstrom")
        ang2bohr = LEN_CONV["Angstrom"]["Bohr"]
        self.assertAlmostEqual(cell1.volume, (edge*ang2bohr)**3)
        self.assertTrue((cell1.lattice == np.identity(3)*edge*ang2bohr).all())


if __name__ == "__main__":
    unittest.main()