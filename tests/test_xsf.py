import unittest
import numpy as np

from tests.common import make_orthorombic_cell
from pbcpy.base import Coord, DirectCell
from pbcpy.atom import Atom
from pbcpy.grid import DirectGrid, ReciprocalGrid
from pbcpy.field import DirectField
from pbcpy.system import System
from pbcpy.formats.xsf import XSF

class TestQEPP(unittest.TestCase):
    
    def test_xsf_write(self):
        print()
        print("*"*50)
        print("Testing format.xsf.XSF write()")
        nr = [50,60,70]
        grid = make_orthorombic_cell(10,12,14,nr=nr,CellClass=DirectGrid)
        ions = [Atom(label="Ne", pos=[0,0,0], cell=grid)]
        griddata_3d = np.random.random(nr)
        field = DirectField(grid=grid, griddata_3d=griddata_3d)
        system = System(ions=ions, cell=grid, field=field)
        res = XSF("tests/test.xsf").write(system)
        self.assertEqual(res, 0)
