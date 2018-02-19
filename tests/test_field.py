import unittest
import numpy as np

from pbcpy.base import DirectCell, ReciprocalCell, Coord
from pbcpy.grid import DirectGrid, ReciprocalGrid
from pbcpy.field import DirectField, ReciprocalField
from pbcpy.constants import LEN_CONV

from tests.common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell, make_triclinic_cell

class TestField(unittest.TestCase):
    
    def test_direct_scalar_field(self):
        print()
        print("*"*50)
        print("Testing DirectField")
        # Test a constant scalar field
        N = 8
        A, B, C = 5, 10, 6
        nr = np.array([A*20, B*20, C*20])
        grid = make_orthorombic_cell(A=A,B=B,C=C,CellClass=DirectGrid, nr=nr, units="Angstrom")
        d = N/grid.volume
        initial_vals = np.ones(nr)*d
        field = DirectField(grid=grid, griddata_3d=initial_vals)
        #print(initial_vals[0,0,:])
        #print(field[0,0,:])
        self.assertTrue(type(field) is DirectField)
        N1 = field.integral()
        self.assertAlmostEqual(N,N1)

        # interpolate up
        field1 = field.get_3dinterpolation(np.array(nr*1.5,dtype=int))
        N1 = field1.integral()
        self.assertAlmostEqual(N,N1)

        # interpolate down
        field2 = field.get_3dinterpolation(nr//2)
        N1 = field2.integral()
        self.assertAlmostEqual(N,N1)

        # fft
        reciprocal_field = field.fft()
        self.assertAlmostEqual(N, reciprocal_field[0,0,0,0])

        # ifft
        field1 = reciprocal_field.ifft(check_real=True)
        N1 = field1.integral()
        self.assertAlmostEqual(N,N1)

    def test_reciprocal_scalar_field(self):
        print()
        print("*"*50)
        print("Testing ReciprocalScalarField")
        print("TODO: Michele, some simple tests for which we have analytic solutions?")
        # TODO: Michele, some simple tests for which we have analytic solutions?
        pass
