import unittest
import numpy as np

from pbcpy.base import DirectCell, ReciprocalCell, Coord
from pbcpy.grid import DirectGrid, ReciprocalGrid
from pbcpy.field import DirectField, ReciprocalField
from pbcpy.constants import LEN_CONV

from tests.common import run_test_orthorombic, run_test_triclinic, make_orthorombic_cell, make_triclinic_cell

class TestField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
          This setUp is in common for all the test cases below, and it's only execuded once
        """
        # Test a constant scalar field
        N = 8
        A, B, C = 5, 10, 6
        nr = np.array([A*20, B*20, C*20])
        grid = make_orthorombic_cell(A=A,B=B,C=C,CellClass=DirectGrid, nr=nr, units="Angstrom")
        d = N/grid.volume
        initial_vals = np.ones(nr)*d
        cls.constant_field = DirectField(grid=grid, griddata_3d=initial_vals,rank=1)
        cls.N = N

    
    def test_direct_field(self):
        print()
        print("*"*50)
        print("Testing DirectField")
        #print(initial_vals[0,0,:])
        #print(field[0,0,:])
        field = self.constant_field
        N = self.N

        self.assertTrue(type(field) is DirectField)
        N1 = field.integral()
        self.assertAlmostEqual(N,N1)

        # fft
        reciprocal_field = field.fft()
        self.assertAlmostEqual(N, reciprocal_field[0,0,0,0])

        # ifft
        field1 = reciprocal_field.ifft(check_real=True)
        N1 = field1.integral()
        self.assertAlmostEqual(N,N1)

        # gradient
        gradient = field.gradient()
        self.assertTrue(isinstance(gradient, DirectField))
        self.assertEqual(field.rank, 1)
        self.assertEqual(gradient.rank, 3)

    def test_direct_field_interpolation(self):
        field = self.constant_field
        nr = field.grid.nr
        # interpolate up
        field1 = field.get_3dinterpolation(np.array(nr*1.5,dtype=int))
        N1 = field1.integral()
        self.assertAlmostEqual(self.N,N1)

        # interpolate down
        field2 = field.get_3dinterpolation(nr//2)
        N1 = field2.integral()
        self.assertAlmostEqual(self.N,N1)

    def test_direct_field_cut(self):
        field = self.constant_field
        nr = field.grid.nr
        x0 = Coord(pos=[0,0,0], cell=field.grid, basis="Crystal")
        r0 = Coord(pos=[1,0,0], cell=field.grid, basis="Crystal")
        field_cut = field.get_cut(origin=x0, r0=r0, nr=nr[0])
        self.assertTrue(np.isclose(field_cut[:,0,0,0], field[:,0,0,0]).all())



    def test_reciprocal_field(self):
        print()
        print("*"*50)
        print("Testing ReciprocalScalarField")
        print("TODO: Michele, some simple tests for which we have analytic solutions?")
        # TODO: Michele, some simple tests for which we have analytic solutions?
        pass





