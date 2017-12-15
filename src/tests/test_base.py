import unittest
import numpy as np
import sys
from env import PBC_SRC
#sys.path.insert(0,"../")
sys.path.insert(0,PBC_SRC)
from pbcpy.base import DirectCell, ReciprocalCell, Coord
from pbcpy.constants import LEN_CONV

class TestDirectCell(unittest.TestCase):
    
    def test_orthorombic_cell(self):
        ## Orthorombic Cell, compare to QE
        A, B, C = 10, 15, 7
        qe_alat = 18.8973
        qe_volume = 7085.7513

        qe_direct = np.zeros((3,3))
        qe_direct[:,0] = (1.000000, 0.000000, 0.000000 )
        qe_direct[:,1] = (0.000000, 1.500000, 0.000000 )
        qe_direct[:,2] = (0.000000, 0.000000, 0.700000 )
        qe_direct *= qe_alat

        qe_reciprocal = np.zeros((3,3))
        qe_reciprocal[:,0] = (1.000000, 0.000000, 0.000000 )
        qe_reciprocal[:,1] = (0.000000, 0.666667, 0.000000 )
        qe_reciprocal[:,2] = (0.000000, 0.000000, 1.428571 )
        qe_reciprocal *= 2*np.pi/qe_alat

        cell = get_orthorombic_cell(A,B,C,"Angstrom")
        
        ang2bohr = LEN_CONV["Angstrom"]["Bohr"]
        self.assertAlmostEqual(cell.volume/qe_volume, 1.)

        ref = qe_direct
        act = cell.lattice
        #print(act)
        self.assertTrue(np.isclose(act,ref).all())

        # ReciprocalCell, check if it matches QE
        reciprocal = cell.get_reciprocal(convention="p")
        ref = qe_reciprocal
        act = reciprocal.lattice
        #print(act)
        self.assertTrue(np.isclose(act,ref).all())
        
        # back to the DirectCell, check if it matches QE
        direct = reciprocal.get_direct(convention="p")
        ref = qe_direct
        act = direct.lattice
        #print(act)
        self.assertTrue(np.isclose(act,ref).all())

    def test_triclinic_cell(self):
        ## Triclinic Cell, compare to QE
        A, B, C = 10, 15, 7
        alpha, beta, gamma = np.pi/3.5, np.pi/2.5, np.pi/3.
        cosAB, cosAC, cosBC = np.cos(gamma), np.cos(beta), np.cos(alpha)
        #print(cosAB, cosAC, cosBC)
        qe_alat = 18.8973
        qe_volume = 4797.6235

        qe_direct = np.zeros((3,3))
        qe_direct[:,0] = (1.000000, 0.000000, 0.000000 )
        qe_direct[:,1] = (0.750000, 1.299038, 0.000000 )
        qe_direct[:,2] = (0.216312, 0.379073, 0.547278 )
        qe_direct *= qe_alat

        qe_reciprocal = np.zeros((3,3))
        qe_reciprocal[:,0] = (1.000000, -0.577350, 0.004652 )
        qe_reciprocal[:,1] = (0.000000, 0.769800, -0.533204 )
        qe_reciprocal[:,2] = (0.000000, 0.000000, 1.827226 )
        qe_reciprocal *= 2*np.pi/qe_alat

        #print(qe_direct)
        #print(qe_reciprocal)
        cell = get_triclinic_cell(A,B,C,alpha,beta,gamma,"Angstrom")
        ang2bohr = LEN_CONV["Angstrom"]["Bohr"]
        self.assertAlmostEqual(cell.volume/qe_volume, 1.)

        ref = qe_direct
        act = cell.lattice
        #print(act)
        self.assertTrue(np.isclose(act,ref,rtol=1.e-5).all())

        # ReciprocalCell, check if it matches QE
        reciprocal = cell.get_reciprocal(convention="p")
        ref = qe_reciprocal
        act = reciprocal.lattice
        #print(act)
        self.assertTrue(np.isclose(act,ref,rtol=1.e-4).all()) # not enough sigfigs in the QE output, increase relative tolerance to 1.e-4
        
        # back to the DirectCell, check if it matches QE
        direct = reciprocal.get_direct(convention="p")
        ref = qe_direct
        act = direct.lattice
        #print(act)
        self.assertTrue(np.isclose(act,ref,rtol=1.e-5).all())

    def test_coord(self):
        # 9x12x18 cell
        cell1 = get_orthorombic_cell(9,12,18,"Angstrom")
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
        

def get_orthorombic_cell(A,B,C, units="Angstrom"):
    lattice = np.identity(3)
    lattice[0,0] = A
    lattice[1,1] = B
    lattice[2,2] = C
    return DirectCell(lattice=lattice, origin=[0,0,0], units=units)

def get_triclinic_cell(A,B,C, alpha, beta, gamma, units="Angstrom"):
    lattice = np.zeros((3,3))
    lattice[:,0] = (A, 0., 0.)
    lattice[:,1] = (B*np.cos(gamma), B*np.sin(gamma), 0.)
    lattice[:,2] = (C*np.cos(beta),
                    C*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
                    C*np.sqrt( 1. + 2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                    - np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2 )/np.sin(gamma)
    )
    return DirectCell(lattice=lattice, origin=[0,0,0], units=units)

if __name__ == "__main__":
    unittest.main()
