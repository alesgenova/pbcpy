import numpy as np
import scipy.special as sp
from .functional_output import Functional
from .local_functionals_utils import vonWeizsackerEnergy, vonWeizsackerPotential
from .local_functionals_utils import ThomasFermiEnergy, ThomasFermiPotential


cTF = 0.3*(3.0 * np.pi**2)**(2.0/3.0)

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
        # cTF = np.pi**2/(3.0 * np.pi**2)**(1.0/3.0) #2.87123400018819

        factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0/3.0))
        tkf = 2.0 * (3.0 * rho0 * np.pi**2)**(1.0/3.0)

        # return (1.2*LindhardFunction(q/tkf,1.0,1.0))*cTF
        # return LindhardFunction(q/tkf,x,y)*factor
        return LindhardFunction2(q/tkf,x,y)*factor

def WTPotential(rho, rho0, Kernel, alpha, beta):
    pot1 = alpha * rho ** (alpha - 1.0) * ((rho ** beta).fft() * Kernel).ifft(force_real = True)
    pot2 = beta * rho ** (beta - 1.0) * ((rho ** alpha).fft() * Kernel).ifft(force_real = True)

    return cTF * (pot1 + pot2)

def WTEnergy(rho, rho0, Kernel, alpha, beta):
    pot1 = ((rho ** beta).fft() * Kernel).ifft(force_real = True)

    return cTF * (rho ** alpha * pot1)


def WT(rho,x=1.0,y=1.0,Sigma=0.025, alpha = 5.0/6.0, beta = 5.0/6.0):
    
    #Only performed once for each grid
    gg = rho.grid.get_reciprocal().gg
    rho0 = np.sum(rho)/np.size(rho)
    KE_kernel = WT_kernel(np.sqrt(gg),rho0, alpha = alpha, beta = beta)

    pot = y*vonWeizsackerPotential(rho,Sigma)+x*ThomasFermiPotential(rho) + WTPotential(rho, rho0, KE_kernel, alpha, beta)
    ene = y*vonWeizsackerEnergy(rho)+ThomasFermiEnergy(rho) + WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    # ene = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    # pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)

    OutFunctional = Functional(name='WT')
    OutFunctional.potential = pot
    OutFunctional.energydensity = ene
    return OutFunctional
