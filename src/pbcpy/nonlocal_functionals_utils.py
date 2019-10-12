import numpy as np
import scipy.special as sp
from scipy.interpolate import splrep
from .functional_output import Functional
from .field import DirectField
# from .local_functionals_utils import vonWeizsackerEnergy, vonWeizsackerPotential
# from .local_functionals_utils import ThomasFermiEnergy, ThomasFermiPotential
from .local_functionals_utils import TF, vW
from .math_utils import multiply, add, power
from scipy.interpolate import interp1d, splrep, splev
from scipy.special import spherical_jn
try:
    from numba import njit, jit
except :
    pass

KE_kernel_saved ={'Kernel':None, 'rho0':0.0, 'shape':None, \
        'KernelTable':None, 'etaMax':None, 'KernelDeriv':None}

def LindhardFunction(eta,lbda,mu):
    '''
    The Inverse Lindhard Function
    
    Attributes
    ----------
    eta: numpy array
    lbda, mu: floats (TF and vW contributions)
    
    '''
    if isinstance(eta, (np.ndarray, np.generic)):
        #
        cond0 = ((eta > 0.0) & (np.abs(eta - 1.0) > 1.0e-10))
        cond1 = eta < 1.0e-10 
        cond2 = np.abs(eta - 1.0) < 1.0e-10
        cond3 = eta > 3.65
        cond11 = eta > 1.0e-10
        
        invEta2 = eta.copy()
        invEta2[cond11] = 1.0 / eta[cond11]**2

        #    
        indx0 = cond0
        indx1 = np.where( cond1 )
        indx2 = np.where( cond2 )
        indx3 = np.where( cond3 )
        
        LindG = eta.copy()
        
        
        LindG[indx0] = 1.0 / (0.5 + 0.25*(1.0-eta[indx0]**2)* 
                    np.log((1.0 + eta[indx0])/np.abs(1.0-eta[indx0]))/eta[indx0])-3.0 * mu * eta[indx0]**2 - lbda

        LindG[indx1] = 1.0 - lbda + eta[indx1]**2 * (1.0 / 3.0 - 3.0 * mu)
        LindG[indx2] = 2.0 - lbda - 3.0 * mu + 20.0 * (eta[indx2]-1.0)
        LindG[indx3] = 3.0*(1.0-mu)*eta[indx3]**2-lbda-0.6   \
                + invEta2[indx3] * (-0.13714285714285712     \
                + invEta2[indx3] * (-6.39999999999999875E-2  \
                + invEta2[indx3] * (-3.77825602968460128E-2  \
                + invEta2[indx3] * (-2.51824061652633074E-2  \
                + invEta2[indx3] * (-1.80879839616166146E-2  \
                + invEta2[indx3] * (-1.36715733124818332E-2  \
                + invEta2[indx3] * (-1.07236045520990083E-2  \
                + invEta2[indx3] * (-8.65192783339199453E-3  \
                + invEta2[indx3] * (-7.1372762502456763E-3   \
                + invEta2[indx3] * (-5.9945117538835746E-3   \
                + invEta2[indx3] * (-5.10997527675418131E-3  \
                + invEta2[indx3] * (-4.41060829979912465E-3  \
                + invEta2[indx3] * (-3.84763737842981233E-3  \
                + invEta2[indx3] * (-3.38745061493813488E-3  \
                + invEta2[indx3] * (-3.00624946457977689E-3) \
                                   ))))))))))))))
        return LindG

def LindhardFunction2(eta,lbda,mu):
    '''
    (1) for x -> 0.0
            2      4  
           x    8⋅x   
       1 + ── + ──── + ... 
           3     45   
    
    (2)  for x -> 1.0  
        2 + (1-x)⋅(2⋅log(1-x) - 2⋅log(2)) + ... 
        We use a magic number 48, because 2.0*(log(1E-10)-log(2))~ -47.4
       
    (3) for y -> 0.0, y = 1/x   
                     2      4          6   
        3    3   24⋅y    8⋅y    12728⋅y   
        ── - ─ - ───── - ──── - ──────── -...
         2   5    175    125     336875   
        y                                                                                                                                                                  
        Actually, if not write the multiplication using C++ or Fortran, numpy.log will be faster.

    The Inverse Lindhard Function
    
    Attributes
    ----------
    eta: numpy array
    lbda, mu: floats (TF and vW contributions)
    
    '''
    if isinstance(eta, (np.ndarray, np.generic)):
        LindG  = np.zeros_like(eta)
        atol = 1.0E-10

        cond0 = np.logical_and(eta > atol, np.abs(eta - 1.0) > atol)
        cond1 = eta < atol
        cond2 = np.abs(eta - 1.0) < atol

        LindG[cond0] = 1.0 / (0.5 + 0.25*(1.0-eta[cond0]**2)* 
                    np.log((1.0 + eta[cond0])/np.abs(1.0-eta[cond0]))/eta[cond0])-3.0 * mu * eta[cond0]**2 - lbda

        LindG[cond1] = 1.0 + eta[cond1]**2 * (1.0 / 3.0 - 3.0 * mu) - lbda
        LindG[cond2] = 2.0 - 48 * np.abs(eta[cond2] - 1.0) - 3.0 * mu * eta[cond2] ** 2 - lbda
        return LindG

def LindhardFunction3(eta,lbda,mu):
    if isinstance(eta, (np.ndarray, np.generic)):
        LindG  = np.zeros_like(eta)
        LindG = (0.5 + 0.25*(eta**2-1.0)* 
                    np.log(np.abs(1.0 - eta)/(1.0+eta))/eta)-3.0 * mu * eta**2 - lbda

        return LindG

def LindhardDerivative(eta, mu):
    LindDeriv  = np.zeros_like(eta)
    atol = 1.0E-10
    cond0 = np.logical_and(eta > atol, np.abs(eta - 1.0) > atol)
    cond1 = eta < atol
    cond2 = np.abs(eta - 1.0) < atol

    TempA = np.log(np.abs((1.0+eta[cond0])/(1.0-eta[cond0])))
    LindDeriv[cond0] = ( 0.5/eta[cond0] - 0.25 * (eta[cond0] ** 2 + 1.0)/eta[cond0] ** 2 * TempA) \
            / (0.5+0.25 * (1 - eta[cond0] ** 2)/eta[cond0] * TempA) ** 2  + \
            6.0 * eta[cond0] * mu
    LindDeriv[cond1] = -2.0 * eta[cond1] * (1.0/3.0 - 3.0 * mu)
    LindDeriv[cond2] = -48

    return LindDeriv * eta

def MGP_kernel(q,rho0,LumpFactor,MaxPoints):
        ''' 
        The MGP Kernel
        '''
        #cTF_WT = 2.87123400018819
        cTF = np.pi**2/(3.0 * np.pi**2)**(1.0/3.0)
        tkf = 2.0 * (3.0 * rho0 * np.pi**2)**(1.0/3.0)
        t_var  = 1.0/(MaxPoints)
        deltat = 1.0/(MaxPoints)
        dt     = deltat / 100

        kertmp = np.zeros(np.shape(q))
        
        for i_var in range(MaxPoints):
            kertmp = kertmp + \
            0.5*((LindhardFunction(q/(tkf*(t_var+dt)**(1.0/3.0)),-0.60,1.0) \
            -LindhardFunction(q/(tkf*(t_var-dt)**(1.0/3.0)),-0.60,1.0))/dt)* \
            t_var**(5.0/6.0)
            #
            t_var = t_var + deltat
            
        tmpker1 = -1.2*kertmp*deltat
        indx    = np.where(q != 0)
        tmpker2 = kertmp.copy()
        tmpker2[indx] = 4*np.pi*sp.erf(q[indx])**2*LumpFactor*np.exp(-q[indx]**2*LumpFactor)/q[indx]**2/cTF
        indx    = np.where(q == 0)
        tmpker2[indx] = q[indx]**2
        tmpker3 = 1.2*LindhardFunction(q/tkf,1.0,1.0) 
        
        return (tmpker1 + tmpker2 + tmpker3)*cTF #*cTF_WT

def WT_kernel(q,rho0, x = 1.0, y = 1.0, alpha = 5.0/6.0, beta = 5.0/6.0):
        ''' 
        The WT Kernel
        '''
        cTF = 0.3*(3.0 * np.pi**2)**(2.0/3.0)
        factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0/3.0))
        tkf = 2.0 * (3.0 * rho0 * np.pi**2)**(1.0/3.0)

        factor *= cTF
        return LindhardFunction2(q/tkf,x,y)*factor

def WTPotential(rho, rho0, Kernel, alpha, beta):
    alphaMinus1 = alpha - 1.0
    betaMinus1 = beta - 1.0
    pot1 = alpha * rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
    # pot1 = DirectField(grid=rho.grid,griddata_3d= power(rho, beta))
    # pot1 = alpha * power(rho, alphaMinus1) * (pot1.fft() * Kernel).ifft(force_real = True)
    if abs(beta - alpha) < 1E-9 :
        pot2 = pot1
    else :
        pot2 = beta * rho ** betaMinus1 * ((rho ** alpha).fft() * Kernel).ifft(force_real = True)
        # pot2 = DirectField(grid=rho.grid,griddata_3d= power(rho, alpha))
        # pot2 = alpha * power(rho, betaMinus1) * (pot2.fft() * Kernel).ifft(force_real = True)

    return pot1 + pot2

def WTEnergy(rho, rho0, Kernel, alpha, beta):
    rhoBeta = rho ** beta
    if abs(beta - alpha) < 1E-9 :
        rhoAlpha = rhoBeta
    else :
        rhoAlpha = rho ** alpha
    pot1 = (rhoBeta.fft() * Kernel).ifft(force_real = True)
    ene = np.einsum('ijkl, ijkl->', pot1, rhoAlpha) * rho.grid.dV

    return ene

def WTStress(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, EnergyPotential=None):
    rho0 = np.sum(rho)/np.size(rho)
    g = rho.grid.get_reciprocal().g
    gg = rho.grid.get_reciprocal().gg
    q = np.sqrt(gg)
    if EnergyPotential is None :
        global KE_kernel_saved
        if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
            print('Re-calculate KE_kernel')
            KE_kernel = WT_kernel(np.sqrt(gg),rho0, alpha = alpha, beta = beta)
            KE_kernel_saved['Kernel'] = KE_kernel
            KE_kernel_saved['rho0'] = rho0
            KE_kernel_saved['shape'] = np.shape(rho)
        else :
            KE_kernel = KE_kernel_saved['Kernel']
        EnergyPotential = Functional(name='WT')
        EnergyPotential.energy = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    mask = rho.grid.get_reciprocal().mask
    factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0/3.0))
    tkf = 2.0 * (3.0 * rho0 * np.pi**2)**(1.0/3.0)
    tkf = float(tkf)
    rhoG_A = (rho ** alpha).fft()/ rho.grid.volume
    rhoG_B = np.conjugate((rho ** beta).fft())/ rho.grid.volume
    DDrho = LindhardDerivative(q/tkf, y) * rhoG_A * rhoG_B
    stress = np.zeros((3, 3))
    gg[0, 0, 0, 0] = 1.0
    mask2 = mask[..., np.newaxis]
    for i in range(3):
        for j in range(i, 3):
            if i == j :
                fac = 1.0/3.0
            else :
                fac = 0.0
            # den = (g[..., i] * g[..., j]/gg[..., 0]-fac)[..., np.newaxis] * DDrho
            den = (g[..., i][mask] * g[..., j][mask]/gg[mask2]-fac) * DDrho[mask2]
            stress[i, j] = (np.einsum('i->', den)).real
    stress *= np.pi ** 2 /(alpha*beta*rho0**(alpha+beta-2)*tkf/2.0)
    for i in range(3):
        stress[i, i] -= 2.0/3.0 * EnergyPotential.energy/rho.grid.volume
    gg[0, 0, 0, 0] = 0.0

    return stress

def WT(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, calcType='Both'):
    global KE_kernel_saved
    #Only performed once for each grid
    gg = rho.grid.get_reciprocal().gg
    rho0 = np.einsum('ijkl -> ', rho) / np.size(rho)
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('Re-calculate KE_kernel')
        KE_kernel = WT_kernel(np.sqrt(gg),rho0, alpha = alpha, beta = beta)
        KE_kernel_saved['Kernel'] = KE_kernel
        KE_kernel_saved['rho0'] = rho0
        KE_kernel_saved['shape'] = np.shape(rho)
    else :
        KE_kernel = KE_kernel_saved['Kernel']


    ene = pot = 0
    if calcType == 'Energy' :
        ene = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    elif calcType == 'Potential' :
        pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)
    else :
        pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)
        if abs(beta - alpha) < 1E-9 :
            ene = np.einsum('ijkl, ijkl->', pot, rho) * rho.grid.dV / (2 * alpha)
        else :
            ene = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    # TF_VW = x_TF_y_vW(rho,x=x,y=y,Sigma=Sigma, calcType = calcType)
    xTF = TF(rho, calcType)
    yvW = vW(rho, Sigma, calcType)
    pot += x * xTF.potential + y * yvW.potential
    ene += x * xTF.energy + y * yvW.energy

    OutFunctional = Functional(name='WT')
    OutFunctional.potential = pot
    OutFunctional.energy= ene
    return OutFunctional

def WT_Kernel_Table(eta, x = 1.0, y = 1.0, alpha = 5.0/6.0, beta = 5.0/6.0):
    '''
    Tip : In this version, this is only work for alpha = beta = 5.0/6.0
    '''
    # factor =1.2*np.pi**2/(3.0*np.pi**2)**(1.0/3.0)
    # factor = 5.0 / (9.0 * alpha * beta) * (0.3 * (3.0  *  np.pi ** 2) ** (2.0/3.0)) * 2 * alpha
    factor =0.4*(3.0*np.pi**2)**(2.0/3.0)
    return LindhardFunction2(eta,x,y)*factor

def WT_Kernel_Deriv_Table(eta, x = 1.0, y = 1.0, alpha = 5.0/6.0, beta = 5.0/6.0):
    factor = 5.0 / (9.0 * alpha * beta)
    cTF = 0.3*(3.0 * np.pi**2)**(2.0/3.0)
    factor *= cTF
    return LindhardDerivative(eta,y)*factor

def LWT_kernel(q, rho0, KernelTable, etaMax = 10.0):
    '''
    Create the LWT kernel for given rho0 and Kernel Table
    '''
    tkf = 2.0*(3.0*np.pi**2*rho0)**(1.0/3.0)
    eta = q/tkf
    Kernel = np.empty_like(q)
    cond0 = eta < etaMax
    cond1 = np.invert(cond0)
    limit = splev(etaMax, KernelTable)
    Kernel[cond0] = splev(eta[cond0], KernelTable)
    Kernel[cond1] = limit
    # Kernel[cond0] = KernelTable(eta[cond0])
    # Kernel[cond1] = KernelTable(etaMax)
    return Kernel

def LWTPotential(rho, KE_kernel_func, nsp = 40, rhoMin = 1E-15, x=1.0,y=1.0, \
        Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0):

    nsp = 3
    Potential = np.empty_like(rho)
    tol = 1E-14
    alphaMinus1 = alpha - 1.0
    betaMinus1 = beta - 1.0
    rhoMax = np.max(rho) + tol
    rhoAve = np.mean(rho)
    if rhoMin is None :
        rhoMin = rhoAve
    # rhoMin = np.mean(rho)
    step = (rhoMax - rhoMin)/(nsp - 1)
    # PotList = []
    # store difference rho0 corresponding Potential
    rho0 = rhoMin
    nr = rho.grid.nr
    q = rho.grid.get_reciprocal().q
    Kernel = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelTable'],\
            etaMax = KE_kernel_func['etaMax'])
    pot1 = rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
    # PotList.append(pot1)
    mask1 = rho < rho0 + tol
    Potential[mask1] = pot1[mask1]
    mask2 = mask1
    nr2 = *nr, 3
    potA = np.empty(nr2)
    potA[..., 0] = pot1[..., 0]
    ip = 0
    for i in range(1, nsp):
        # print('i', i)
        rho0 += step
        ip += 1
        Kernel = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelTable'],\
                etaMax = KE_kernel_func['etaMax'])
        pot1 = rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
        if ip < 3 :
            potA[..., ip] = pot1[..., 0]
            rhoD = [rho0 - i * step for i in range(ip, -1, -1)]
        else :
            potA[..., 0] = potA[..., 1]
            potA[..., 1] = potA[..., 2]
            potA[..., 2] = pot1[..., 0]
            rhoD = [rho0 - 2 * step, rho0 - step, rho0]
        mask1 = np.invert(mask2)
        mask2 = rho < rho0
        mask = np.logical_and(mask1, mask2)
        # Potential[mask] = potA[mask[..., 0], 1]
        if ip < 3 :
            ib = ip + 1
        else :
            ib = 3
        # Potential = PotentialSpline(rhoD, potA, rho, Potential, mask, ib = 3)
        for i0 in range(nr[0]):
            for i1 in range(nr[1]):
                for i2 in range(nr[2]):
                    if mask[i0, i1, i2] :
                        f = interp1d(rhoD, potA[i0, i1, i2, :ib], kind = 1)
                        Potential[i0, i1, i2] = f(rho[i0, i1, i2])

    # Kernel = LWT_kernel(q, rhoAve, KernelTable = KE_kernel_func['KernelTable'],\
            # etaMax = KE_kernel_func['etaMax'])
    # Potential = rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
    return DirectField(grid=rho.grid,griddata_3d = Potential)

def LWTPotential2(rho, KE_kernel_func, nsp = 40, rhoMin = 1E-10, x=1.0,y=1.0, \
        Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0):

    Potential = np.empty_like(rho)
    tol = 1E-14
    alphaMinus1 = alpha - 1.0
    betaMinus1 = beta - 1.0
    rhoMax = np.max(rho) + tol
    rhoAve = np.mean(rho)
    if rhoMin is None :
        rhoMin = rhoAve
    rhoMin = rhoAve
    step = (rhoMax - rhoMin)/(nsp - 1)
    nr = rho.grid.nr
    q = rho.grid.get_reciprocal().q
    nr2 = *nr, 3
    potA = np.empty(nr2)
    #-----------------------------------------------------------------------
    rho0 = rhoMin
    Kernel = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelTable'],\
            etaMax = KE_kernel_func['etaMax'])
    pot1 = rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
    mask1 = rho < rho0 + tol
    Potential[mask1] = pot1[mask1]
    Potential[rho < tol] = 0.0
    mask2 = mask1
    #-----------------------------------------------------------------------
    potA[..., 0] = pot1[..., 0]
    ip = 0
    for i in range(1, nsp):
        rho0 += step
        ip += 1
        Kernel = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelTable'],\
                etaMax = KE_kernel_func['etaMax'])
        pot1 = rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
        if ip < 3 :
            potA[..., ip] = pot1[..., 0]
            rhoD = [rho0 - i * step for i in range(ip, -1, -1)]
        else :
            potA[..., 0] = potA[..., 1]
            potA[..., 1] = potA[..., 2]
            potA[..., 2] = pot1[..., 0]
            rhoD = [rho0 - 2 * step, rho0 - step, rho0]
        mask1 = np.invert(mask2)
        mask2 = rho < rho0
        mask = np.logical_and(mask1, mask2)
        if not np.any(mask) : continue
        potIni = potA[mask[..., 0], :]
        potSpline = np.empty(potIni.shape[:-1])
        rhoL = rho[mask]
        if ip < 3 :
            ib = ip + 1
        else :
            ib = 3
        k = ib - 1
        for i in range(np.size(potSpline)):
            f = interp1d(rhoD, potIni[i, :ib], kind = k)
            potSpline[i] = f(rhoL[i])
        Potential[mask] = potSpline

    # Kernel = LWT_kernel(q, rhoAve, KernelTable = KE_kernel_func['KernelTable'],\
            # etaMax = KE_kernel_func['etaMax'])
    # Potential = rho ** alphaMinus1 * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
    return DirectField(grid=rho.grid,griddata_3d = Potential)

def LWTPotentialEnergy(rho, KE_kernel_func, nsp = 40, rhoMin = 1E-10, x=1.0,y=1.0, \
        Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0, calcType = 'Both'):

    Potential = np.empty_like(rho)
    tol = 1E-14
    alphaMinus1 = alpha - 1.0
    betaMinus1 = beta - 1.0
    rhoMax = np.max(rho) + tol
    rhoAve = np.mean(rho)
    if rhoMin is None :
        rhoMin = rhoAve
    rhoMin = rhoAve
    step = (rhoMax - rhoMin)/(nsp - 1)
    nr = rho.grid.nr
    q = rho.grid.get_reciprocal().q
    nr2 = *nr, 3
    potA = np.empty(nr2)
    potA[..., 0] = 0.0
    rhoAlpha1 = rho ** alphaMinus1
    rhoBeta = rho ** beta
    rhoBetaG = rhoBeta.fft()
    #-----------------------------------------------------------------------
    rho0 = rhoMin * 0.5
    Kernel = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelTable'],\
            etaMax = KE_kernel_func['etaMax'])
    pot1 = rhoAlpha1 * (rhoBetaG * Kernel).ifft(force_real = True)
    potA[..., 1] = pot1[..., 0]
    rho0 = rhoMin
    Kernel = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelTable'],\
            etaMax = KE_kernel_func['etaMax'])
    pot1 = rhoAlpha1 * (rhoBetaG * Kernel).ifft(force_real = True)
    potA[..., 2] = pot1[..., 0]
    mask1 = rho < rho0
    mask = mask1
    rhoD = [0.0, 0.5 * rho0, rho0]
    potIni = potA[mask[..., 0], :]
    potSpline = np.empty(potIni.shape[:-1])
    rhoL = rho[mask]
    Potential[mask] = potA[mask[..., 0], 1]
    # for i in range(np.size(potSpline)):
        # f = interp1d(rhoD, potIni[i, :], kind = 2)
        # potSpline[i] = f(rhoL[i])
    # Potential[mask] = potSpline
    mask2 = mask1
    #-----------------------------------------------------------------------
    ip = 0
    for i in range(1, nsp):
        rho0 += step
        ip += 1
        Kernel = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelTable'],\
                etaMax = KE_kernel_func['etaMax'])
        pot1 = rhoAlpha1 * (rhoBetaG * Kernel).ifft(force_real = True)
        potA[..., 0] = potA[..., 1]
        potA[..., 1] = potA[..., 2]
        potA[..., 2] = pot1[..., 0]
        if ip  == 1 :
            rhoD = [(rho0 - step) * 0.5, rho0 - step, rho0]
        else :
            rhoD = [rho0 - 2 * step, rho0 - step, rho0]
        mask1 = np.invert(mask2)
        mask2 = rho < rho0
        mask = np.logical_and(mask1, mask2)
        if not np.any(mask) : continue
        potIni = potA[mask[..., 0], :]
        potSpline = np.empty(potIni.shape[:-1])
        rhoL = rho[mask]
        for i in range(np.size(potSpline)):
            f = interp1d(rhoD, potIni[i, :], kind = 2)
            potSpline[i] = f(rhoL[i])
        Potential[mask] = potSpline

    ############################## Kernel Part ##############################
    OutFunctional = Functional(name='LWT')
    OutFunctional.potential = 0.0
    OutFunctional.energy= 0.0
    if calcType == 'Energy' or calcType == 'Both' :
        ene = 3.0/5.0 * np.einsum('ijkl, ijkl ->', Potential, rho) * rho.grid.dV
    if calcType == 'Potential' or calcType == 'Both' :
        if abs(beta - alpha) < 1E-9 :
            rhoAlpha = rhoBeta
        else :
            rhoAlpha = rho ** alpha
        rho0 = np.mean(rho)
        KernelDeriv = LWT_kernel(q, rho0, KernelTable = KE_kernel_func['KernelDeriv'],\
                etaMax = KE_kernel_func['etaMax'])
        pot1 = 1.0/3 * rhoAlpha1 * ((rhoBetaG * KernelDeriv).ifft(force_real = True))
        Potential += pot1
        pot = DirectField(grid=rho.grid,griddata_3d = Potential)
        OutFunctional.potential = pot
    return OutFunctional

def LWT(rho, nsp = 40, rhoMin = 1E-15, x=1.0,y=1.0, Sigma=0.025, \
        alpha = 5.0/6.0, beta = 5.0/6.0, etaMax = 10, Neta = 4000, order = 3, calcType='Both'):
    global KE_kernel_saved
    etaMax = 40.0
    #Only performed once for each grid
    gg = rho.grid.get_reciprocal().gg
    rho0 = np.mean(rho)
    if abs(KE_kernel_saved['rho0']-rho0) > 1E-6 or np.shape(rho) != KE_kernel_saved['shape'] :
        print('rho0', rho0)
        print('Re-calculate KernelTable')
        eta = np.linspace(0, etaMax, Neta)
        KernelTable = WT_Kernel_Table(eta, x, y, alpha, beta)
        KernelDerivTable = WT_Kernel_Deriv_Table(eta, x, y, alpha, beta)
        KE_kernel_saved['KernelTable'] = splrep(eta, KernelTable, k=order)
        KE_kernel_saved['KernelDeriv'] = splrep(eta, KernelDerivTable, k=order)
        KE_kernel_saved['etaMax'] = etaMax
        KE_kernel_saved['shape'] = np.shape(rho)
        KE_kernel_saved['rho0'] = rho0
    KE_kernel_func = KE_kernel_saved

    # pot = LWTPotential(rho, KE_kernel_func, nsp, rhoMin, x, y, Sigma, alpha, beta)
    # pot = LWTPotential3(rho, KE_kernel_func, nsp, rhoMin, x, y, Sigma, alpha, beta)
    # if calcType == 'Energy' :
        # ene = 3.0/5.0 * np.einsum('ijkl, ijkl ->', pot, rho) * rho.grid.dV
        # pot = 0
    # elif calcType == 'Potential' :
        # ene = 0
    # else :
        # ene = 3.0/5.0 * np.einsum('ijkl, ijkl ->', pot, rho) * rho.grid.dV
        
    # TF_VW = x_TF_y_vW(rho,x=x,y=y,Sigma=Sigma, calcType = calcType)
    OutFunctional = LWTPotentialEnergy(rho, KE_kernel_func, nsp, \
            rhoMin, x, y, Sigma, alpha, beta, calcType = calcType)
    xTF = TF(rho, calcType)
    yvW = vW(rho, Sigma, calcType)
    OutFunctional.potential += x * xTF.potential + y * yvW.potential
    OutFunctional.energy += x * xTF.energy + y * yvW.energy
    return OutFunctional
