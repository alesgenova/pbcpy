import numpy as np
from pbcpy.constants import LEN_CONV
## Orthorombic Cell, compare to QE
def run_test_orthorombic(self, cell_cls, nr=None):
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

    ang2bohr = LEN_CONV["Angstrom"]["Bohr"]
    #print("cell")
    cell = make_orthorombic_cell(A,B,C,CellClass=cell_cls,nr=nr,units="Angstrom")
    #print("cell1")
    cell1 = make_orthorombic_cell(A*ang2bohr,B*ang2bohr,C*ang2bohr,CellClass=cell_cls,nr=nr,units="Bohr")
    #print()
    #print(cell.lattice)
    #print(cell1.lattice)
    self.assertTrue(cell==cell)
    self.assertTrue(cell==cell1)
    self.assertAlmostEqual(cell.volume/qe_volume, 1.)

    ref = qe_direct
    act = cell.lattice
    #print(act)
    self.assertTrue(np.isclose(act,ref).all())

    # ReciprocalCell, check if it matches QE
    #print("reciprocal")
    reciprocal = cell.get_reciprocal(convention="p")
    # can't compare == a reciprocal and a direct cell
    with self.assertRaises(TypeError):
        is_same = cell == reciprocal

    ref = qe_reciprocal
    act = reciprocal.lattice
    #print(act)
    self.assertTrue(np.isclose(act,ref).all())
    
    # back to the DirectCell, check if it matches QE
    #print("direct")
    direct = reciprocal.get_direct(convention="p")
    ref = qe_direct
    act = direct.lattice
    #print(act)
    self.assertTrue(np.isclose(act,ref).all())

def run_test_triclinic(self, cell_cls, nr=None):
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
    ang2bohr = LEN_CONV["Angstrom"]["Bohr"]
    #print()
    cell = make_triclinic_cell(A,B,C,alpha,beta,gamma,CellClass=cell_cls,nr=nr,units="Angstrom")
    #print()
    cell1 = make_triclinic_cell(A*ang2bohr,B*ang2bohr,C*ang2bohr,alpha,beta,gamma,CellClass=cell_cls,nr=nr,units="Bohr")
    #print()
    self.assertTrue(cell==cell1)
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

def make_orthorombic_cell(A,B,C, CellClass, nr=None, units="Angstrom"):
    lattice = np.identity(3)
    lattice[0,0] = A
    lattice[1,1] = B
    lattice[2,2] = C
    lattice *= LEN_CONV[units]["Bohr"] # pbcpy doesn't have units for the time being, always feed it Bohr
    if nr is None:
        return CellClass(lattice=lattice, origin=[0,0,0], units=None)
    else:
        return CellClass(lattice=lattice, nr=nr, origin=[0,0,0], units=None)

def make_triclinic_cell(A,B,C, alpha, beta, gamma, CellClass, nr=None, units="Angstrom"):
    lattice = np.zeros((3,3))
    lattice[:,0] = (A, 0., 0.)
    lattice[:,1] = (B*np.cos(gamma), B*np.sin(gamma), 0.)
    lattice[:,2] = (C*np.cos(beta),
                    C*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
                    C*np.sqrt( 1. + 2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                    - np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2 )/np.sin(gamma)
    )
    lattice *= LEN_CONV[units]["Bohr"] # pbcpy doesn't have units for the time being, always feed it Bohr
    if nr is None:
        return CellClass(lattice=lattice, origin=[0,0,0], units=None)
    else:
        return CellClass(lattice=lattice, nr=nr, origin=[0,0,0], units=None)