import numpy as np
from .atom import Atom
from .field import DirectField

def NuclearElectron(ions,density,PPs, calcType='Both'):
    '''Computes the local part of the PP
    Input: ions (coord), density (rank-0 PBCpy field), PPs (array of str)'''
    if len(PPs) != len(ions.Zval):
        raise ValueError("Incorrect number of pseudopotential files")
    if not isinstance(ions,(Atom)):
        raise AttributeError("Ions must be an array of PBCpy Atom")
    if not isinstance(density,(DirectField)):
        raise AttributeError("Density must be a PBCpy DirectField")
    NuclearElectron = ions.local_PP(grid=density.grid,rho=density,PP_file=PPs, calcType=calcType)
    NuclearElectron.name = 'Local Pseudopotential'
    return NuclearElectron

def NuclearElectronStress(ions,rho,EnergyPotential=None, PP_file=None):
    if EnergyPotential is None :
        EnergyPotential = NuclearElectron(ions, rho, PP_file, calcType='Energy')
    reciprocal_grid=rho.grid.get_reciprocal()
    g= reciprocal_grid.g
    gg= reciprocal_grid.gg
    mask = reciprocal_grid.mask
    mask2 = mask[..., np.newaxis]
    q = np.sqrt(gg)
    q[0, 0, 0, 0] = 1.0
    rhoG = rho.fft()
    stress = np.zeros((3, 3))
    v_deriv=ions.Get_PP_Derivative(rho.grid)
    rhoGV_q = rhoG * v_deriv / q
    for i in range(3):
        for j in range(i, 3):
            # den = (g[..., i]*g[..., j])[..., np.newaxis] * rhoGV_q
            # stress[i, j] = (np.einsum('ijkl->', den)).real / rho.grid.volume
            den = (g[..., i][mask]*g[..., j][mask]) * rhoGV_q[mask2]
            stress[i, j] = -(np.einsum('i->', den)).real / rho.grid.volume*2.0
            if i == j :
                stress[i, j] -= EnergyPotential.energy
    stress /= rho.grid.volume
    return stress

def NuclearElectronForce(ions,rho,PP_file=None):
    rhoG = rho.fft()
    reciprocal_grid = rho.grid.get_reciprocal()
    g = reciprocal_grid.g
    Forces= np.zeros((ions.nat, 3))
    mask = reciprocal_grid.mask
    mask2 = mask[..., np.newaxis]
    # for i in range(ions.nat):
        # strf = ions.istrf(reciprocal_grid, i)
        # Forces[i] = np.einsum('ijkl,ijkl->l', reciprocal_grid.g, \
                # ions.vlines[ions.labels[i]]* (rhoG * strf).imag)
    # Forces /= rho.grid.volume
    for i in range(ions.nat):
        strf = ions.istrf(reciprocal_grid, i)
        den = ions.vlines[ions.labels[i]][mask2]* (rhoG[mask2] * strf[mask2]).imag
        for j in range(3):
            Forces[i, j] = np.einsum('i, i->', reciprocal_grid.g[..., j][mask], den)
    Forces *= 2.0/rho.grid.volume
    return Forces

def NuclearElectronForcePME(ions,rho,PP_file=None):
    rhoG = rho.fft()
    reciprocal_grid = rho.grid.get_reciprocal()
    g = reciprocal_grid.g
    gg = rho.grid.get_reciprocal().gg
    # mask = reciprocal_grid.mask
    # mask2 = mask[..., np.newaxis]
    Bspline = ions.Bspline
    Barray = Bspline.Barray
    Barray = np.conjugate(Barray)
    denG = rhoG * Barray
    nr = rho.grid.nr

    cell_inv = np.linalg.inv(ions.pos[0].cell.lattice)
    Forces= np.zeros((ions.nat, 3))
    ixyzA = np.mgrid[:ions.BsplineOrder, :ions.BsplineOrder, :ions.BsplineOrder].reshape((3, -1))
    Q_derivativeA = np.zeros((3, ions.BsplineOrder * ions.BsplineOrder * ions.BsplineOrder))
    for key in ions.Zval.keys():
        denGV = denG * ions.vlines[key]
        denGV[0, 0, 0, 0] = 0.0+0.0j
        rhoPB = denGV.ifft(force_real = True)[..., 0]
        for i in range(ions.nat):
            if ions.labels[i] == key :
                Up = np.array(ions.pos[i].to_crys()) * nr
                Mn = []
                Mn_2 = []
                for j in range(3):
                    Mn.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j])) )
                    Mn_2.append( Bspline.calc_Mn(Up[j] - np.floor(Up[j]), order = ions.BsplineOrder - 1) )
                Q_derivativeA[0] = nr[0] * np.einsum('i, j, k -> ijk', Mn_2[0][1:]-Mn_2[0][:-1], Mn[1][1:], Mn[2][1:]).reshape(-1)
                Q_derivativeA[1] = nr[1] * np.einsum('i, j, k -> ijk', Mn[0][1:], Mn_2[1][1:]-Mn_2[1][:-1], Mn[2][1:]).reshape(-1)
                Q_derivativeA[2] = nr[2] * np.einsum('i, j, k -> ijk', Mn[0][1:], Mn[1][1:], Mn_2[2][1:]-Mn_2[2][:-1]).reshape(-1)
                l123A = np.mod(1+np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nr.reshape((3, 1)))
                Forces[i] = -np.sum(np.matmul(Q_derivativeA.T, cell_inv) * rhoPB[l123A[0], l123A[1], l123A[2]][:, np.newaxis], axis=0)
    return Forces

def NuclearElectronStressPME(ions,rho,EnergyPotential=None, PP_file=None):
    if EnergyPotential is None :
        EnergyPotential = NuclearElectron(ions, rho, PP_file, calcType='Energy')
    rhoG = rho.fft()
    reciprocal_grid = rho.grid.get_reciprocal()
    g = reciprocal_grid.g
    gg = rho.grid.get_reciprocal().gg
    q = np.sqrt(gg)
    q[0, 0, 0, 0] = 1.0
    mask = reciprocal_grid.mask
    mask2 = mask[..., np.newaxis]
    Bspline = ions.Bspline
    Barray = Bspline.Barray
    rhoGB = np.conjugate(rhoG) * Barray
    nr = rho.grid.nr
    stress = np.zeros((3, 3))
    QA = np.empty(nr)
    for key in ions.Zval.keys():
        rhoGBV = rhoGB * ions.Get_PP_Derivative_One(rho.grid, key = key)
        # Qarray = DirectField(grid=rho.grid,griddata_3d=np.zeros_like(q), rank=1)
        QA[:] = 0.0
        for i in range(ions.nat):
            if ions.labels[i] == key :
                # Qarray += ions.Bspline.get_PME_Qarray(i)
                QA = ions.Bspline.get_PME_Qarray(i, QA)
        Qarray = DirectField(grid=rho.grid,griddata_3d=QA, rank=1)
        rhoGBV = rhoGBV * (Qarray.fft())
        for i in range(3):
            for j in range(i, 3):
                den = (g[..., i][mask]*g[..., j][mask]) * rhoGBV[mask2]/ q[mask2]
                # stress[i, j] -= (np.einsum('i->', den)).real / 
                # stress[i, j] -= (np.einsum('i->', den)).real * rho.grid.dV
                stress[i, j] -= (np.einsum('i->', den)).real / rho.grid.volume**2
    stress *= 2.0 * rho.grid.nnr
    for i in range(3):
        stress[i, i] -= EnergyPotential.energy
    stress /= rho.grid.volume
    return stress
