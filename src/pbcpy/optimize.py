import numpy as np
from scipy.optimize import minimize
from .field import DirectField, ReciprocalField
from .hartree import HartreeFunctional
from .formats.qepp import PP
from .local_pseudopotential import NuclearElectron
from .ewald import ewald

class optimize(object):

    def __init__(self, ppfile=None, ions=None, rho=None, KEDF=None, ExchangeCorrelation=None, PP=None, 
                 initial_guess=None, 
                 optimization_method=None, 
                 method_options=None, 
                 verbose=False):
        '''
        This class minimizes the energy varying a DirectField rho while maintaining
        constant the number of electrons. 
        $$
         E[\rho] \to E[\frac{N}{\int \rho}\rho].
        $$
        INPUT: ppfile      a pp file from some other calculation (QE, pbcpy, etc..).
               ions        Atom class list.
               rho         DirectField, the electron density.
               KEDF        the Kinetic energy functional
               XC          the XC functional
               PP          list of pseudopotential files.
               verbose     optional, wanna print stuff?
        '''

        self.verbose = verbose
        
        self.res_ = None

        if ions is not None:
            if not isinstance(ions,list):
                raise AttributeError("ions must be a list of Atoms")
            self.ions      = ions
        else:
            raise Exception("Must pass ions to Optimize")
            
        if PP is not None:
            if not isinstance(PP,list):
                raise AttributeError("PP must be a list of pp files")
            self.PP      = PP
        else:
            raise Exception("Must pass pseudo potential files to Optimize")

        if rho is not None:
            if not isinstance(rho,DirectField):
                raise AttributeError("rho must be a PBCpy DirectField")
            self.rho       = rho
        else:
            raise Warning("Density rho not passed to Optimize. Try to pass a ppfile.")

        if optimization_method is not None:
            if not isinstance(optimization_method,str):
                raise AttributeError("optimization_method must be a string")
            self.optimization_method       = optimization_method
        else:
            self.optimization_method='L-BFGS-B'
            raise Warning("Optimization method set to L-BFGS-B")

        if method_options is not None:
            self.method_options       = method_options
            if not isinstance(method_options,dict):
                raise AttributeError("method_options must be a dictionary")
        else:
            self.method_options = {}

            
        if ppfile is not None:
            if not isinstance(ppfile,str):
                raise AttributeError("ppfile must be a string")
            self.ppfile       = ppfile
        else:
            raise Warning("ppfile rho not passed to Optimize. Must pass a rho & ions.")

        if self.ppfile is None and rho is None and ions is None:
            raise Exception("either ppfile or rho & ions must be given to Optimize")

        if initial_guess is not None:
            # this will include cases here - not coded yet
            self.initial_guess       = initial_guess
        else:
            raise Warning("Using input rho as initial guess.")

        if KEDF is not None:
            self.KEDF       = KEDF
        else:
            raise Warning("Must pass a KEDF to Optimize")
            
        if ExchangeCorrelation is not None:
            self.ExchangeCorrelation       = ExchangeCorrelation
        else:
            raise Warning("Must pass an Exchange-Correlation functional to Optimize")
            

    def EnergyDensityAndPotential(rho,KEDF,ExchangeCorrelation,ions,PP):
        '''rho: real-space density
        ions: collection of Atoms'''
        EeN                 = NuclearElectron(ions,rho,PP)
        Hartree             = HartreeFunctional(rho)
        EnergyDensity     = KEDF.energydensity-EeN.energydensity+Hartree.energydensity+ExchangeCorrelation.energydensity
        Potential         = KEDF.potential + ExchangeCorrelation.potential + Hartree.potential - EeN.potential
        return EnergyDensity, Potential

    def E_v_phi_ravel(phi,rho,KEDF,ExchangeCorrelation,ions,PP):
        phi_ = DirectField(rho.grid,griddata_3d=np.reshape(phi,np.shape(rho)),rank=1) # I suspect this takes time...
        rho_ = phi_*phi_
        N_=rho_.integral()
        rho_ *= N/N_
        Edens, v_ = EnergyDensityAndPotential(rho_,KEDF,ExchangeCorrelation,ions,PP) # of course, this takes time as well
        E=Edens.integral()
        #print("E = ",E)
        int_tem_ = phi_*phi_*v_
        other_term_ = - int_tem_.integral() / N_
        the_v_ =  v_  
        final_v_ = ( the_v_ + other_term_ ) * 2.0 * phi_  * N/N_ * grid.dV
        return E , final_v_.ravel()

    def optimize(self):
        if self.initial_guess == "constant":
            print("Using a constant density as initial guess.")
            x0=np.zeros_like(self.rho)
            x0[:,:,:,:]=np.sqrt(self.rho.integral() / self.rho.grid.Volume)
            x0 = x0.ravel()
        else:
            x0=self.rho.ravel()
        res = minimize(fun=E_v_phi_ravel,
                       args=(self.rho,self.KEDF,self.ExchangeCorrelation,self.ions,self.PP),
                       jac=True,
                       x0=x0,
                       method=self.optimization_method,
                       options=self.method_options)
        self.res_ = res
        return DirectField(self.rho.grid,griddata_3d=np.reshape(res.x**2,np.shape(self.rho)),rank=1)

