import numpy as np
import scipy.special as sp 
from .Kernel import KF
from scipy.interpolate import interp1d

def LWT_kernel(q,rho,KF,nsp=40,rhomax=1.0):
    '''
    Create the LWT kernel for given Kernel Table.
    '''
    factor =1.2*np.pi**2/(3.0*np.pi**2)**1.0/3.0
    drho = rhomax/float(nsp-1.0)

    print('The rhomax for spline is :        ',rhomax)
    print('The number of rho0 for spline is :',nsp)
    print('The drho for spline is :          ',drho)

    f,eta_min,eta_max = KF().readkernl()

    rho_d = np.zeros(nsp,dtype=float)
    tkf = np.zeros(nsp,dtype=float)
    nq = np.size(q)
    tem_q = np.reshape(q,(-1))
    kernel=np.zeros(nsp,nq)

    for i in range(1,nsp):
        rho_d[i] = drho*i
        tkf[i] = 2.0*(3.0*np.pi**2*rho_d[i])**(1.0/3.0)

        for j in range(0,nq):
            cond=tem_q[j]/tkf[i]
            if (cond <= eta_min):
                kernel[i,j] = -factor*8.0/3.0*cond**2
            elif (cond >= (eta_max-0.5)):
                #kernel[i,j] = factor*(1.0/(0.5+0.25*(1-cond**2)/cond*np.log((1.0+cond)/cond-1.0))-3.0*cond**2-1.0)
                 kernel[i,j] = -6.12531
            else:
                kernel[i,j] = f(cond)
            kernel=np.reshape(kernel,(nsp,q.shape[0],q.shape[1],q.shape[2]))
    return kernel

def LMGP0_kernel(q,rho,KF,nsp=40,rhomax=1.0):
    '''
    Create the LMGP0 kernel for given Kernel Table.
    '''
    factor =1.2*np.pi**2/(3.0*np.pi**2)**1.0/3.0
    drho = rhomax/float(nsp-1.0)

    print('The rhomax for spline is :        ',rhomax)
    print('The number of rho0 for spline is :',nsp)
    print('The drho for spline is :          ',drho)

    f,eta_min,eta_max = KF().readkernl()

    rho_d = np.zeros(nsp,dtype=float)
    tkf = np.zeros(nsp,dtype=float)
    nq = np.size(q)
    tem_q = np.reshape(q,(-1))
    kernel=np.zeros(nsp,nq)

    for i in range(1,nsp):
        rho_d[i] = drho*i
        tkf[i] = 2.0*(3.0*np.pi**2*rho_d[i])**(1.0/3.0)

        for j in range(0,nq):
            cond=tem_q[j]/tkf[i]
            if (cond <= eta_min):
                kernel[i,j] = -factor*8.0/3.0*cond**2
            elif (cond >= (eta_max-0.5)):
                #kernel[i,j] = factor*(1.0/(0.5+0.25*(1-cond**2)/cond*np.log((1.0+cond)/cond-1.0))-3.0*cond**2-1.0)
                 kernel[i,j] = -6.12531
            else:
                kernel[i,j] = f(cond)
            kernel=np.reshape(kernel,(nsp,q.shape[0],q.shape[1],q.shape[2]))
    return kernel

def LMGP_kernel(q,rho,NE,KF,rhomax=1.0,nsp=40,LMGPA=0.2):
    '''
    Create the LMGP kernel for given Kernel Table.
	Input:  q,   qTable
	        rho, electron density 
			NE,  Number of Electron in the system
			KF,  Kenerl Table class
			nsp, number of rho0 for spline (optinal)
			LMGPA, a parameter for moduleing the Kinetic electron
   	Return: The Kernel 	    
    '''
    factor =1.2*np.pi**2/(3.0*np.pi**2)**1.0/3.0
    drho = rhomax/float(nsp-1.0)

    print('The rhomax for spline is :        ',rhomax)
    print('The number of rho0 for spline is :',nsp)
    print('The drho for spline is :          ',drho)

    f,eta_min,eta_max = KF().readkernl()

    rho_d = np.zeros(nsp,dtype=float)
    tkf = np.zeros(nsp,dtype=float)
    nq = np.size(q)
    tem_q = np.reshape(q,(-1))
    kernel=np.zeros(nsp,nq)
    KEker=np.zeros(nq)
    a=LMGPA/NE**(2.0/3.0)
	
    indx  = np.where(tem_q != 0)
    KEker[indx] = 4*np.pi*sp.erf(tem_q[indx])**2*a*np.exp(-tem_q[indx]**2*a)/tem_q[indx]**2
	
    for i in range(1,nsp):
        rho_d[i] = drho*i
        tkf[i] = 2.0*(3.0*np.pi**2*rho_d[i])**(1.0/3.0)
        
        for j in range(0,nq):
            cond=tem_q[j]/tkf[i]
            if (cond <= eta_min):
                kernel[i,j] = -factor*8.0/3.0*cond**2
            elif (cond >= (eta_max-0.5)):
                #kernel[i,j] = factor*(1.0/(0.5+0.25*(1-cond**2)/cond*np.log((1.0+cond)/cond-1.0))-3.0*cond**2-1.0)
                 kernel[i,j] = -6.12531
            else:
                kernel[i,j] = f(cond)
		    
    	kernel[i,:]=kernel[i,:]+KEker[:]       
	kernel=np.reshape(kernel,(nsp,q.shape[0],q.shape[1],q.shape[2]))
    return kernel

def LWTPotEnegy(rho,kernel,CalEnergy=0,nsp=40,rhomax=1.0):
    '''
    Evaluation of the EnergyDensity and Potential

	**LWT, LMGP0,and LMGP can share the same function to evaluate the potential and EnergyDensity**

    Input : rho         DirectField, the electron density
            kernel      narry[nsp,q.sharp[0]],q.sharp[1],q.sharp[2], the kernel
            CalEnergy   int, if calculate Energy density CalEnergy=1
            nsp         int, the number
    Output : Potential
             EnergyDensity
    '''
    fs=5.0/6.0
    mos=-1.0/6.0
    pot=np.zeros((nsp,rho.shape[0],rho.shape[1],rho.shape[2]),dtype=float)
    potential=np.zeros(np.shape(rho),dtype=float)
    EnergyDensity=np.zeros(np.shape(rho),dtype=float)
    drho=rhomax/(nsp-1.0)
    rho_d =np.zeros(nsp,dtype=float)
    for i in range(1,nsp):
        rho_d[i] = rhomax/(nsp-1.00)*i
        pot[i,:,:,:]=rho**mos*(((rho **fs).fft() * Kernel[i,:,:,:]).ifft(force_real = True))

    #spline the point
    for i in range(0,rho.shape[0]):
        for j in range(0,rho.shape[1]):
            for k in range(0,tho.shape[2]):
                indx = int(rho[i,j,k]/drho)
                nb = indx-1
                ne = indx+2
                if (indx < 2 ):
                    f=interp1d(rho_d[0:3],pot[0:3,i,j,k],kind=1)
                    potential[i,j,k]=f(rho[i,j,k])
                elif (indx >= nsp-2):
                    potential[i,j,k]= pot[nsp-2,i,j,k] + (pot[nsp-1,i,j,k]-pot[nsp-2,i,j,k])/drho*(rho[i,j,k]-rho_d[nsp-2])
                else:
                    f=interp1d(rho_d[nb:ne],pot[nb:ne],kind=2)
                    potential[i,j,k]=f(rho[i,j,k])

    if (CalEnergy==1):
        EnergyDensity = 3.0/5.0*potential*rho

    return potential,EnergyDensity
