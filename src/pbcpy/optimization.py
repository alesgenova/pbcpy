import numpy as np
from scipy.optimize import minimize
from .field import DirectField

class Optimization(object):
    

    def __init__(self, 
                 optimization_method='L-BFGS-B', 
                 optimization_options=None,
                 EnergyEvaluator=None,
                 guess_rho=None):
                    
        self.rho = guess_rho

        if optimization_options is None:
            self.optimization_options={}
            self.optimization_options["disp"] = None
            self.optimization_options["maxcor"] = 20
            self.optimization_options["ftol"] = 1.0e-7
            self.optimization_options["gtol"] = 1.0e-7
            self.optimization_options["maxfun"] = 1000
            self.optimization_options["maxiter"] = 100
            self.optimization_options["maxls"] = 10
        else:
            self.optimization_options = optimization_options
        
        if EnergyEvaluator is None:
            raise AttributeError('Must provide an energy evaluator')
        else:
            self.EnergyEvaluator = EnergyEvaluator
        
        self.optimization_method = optimization_method
        
    def get_optimal_rho(self,guess_rho=None):
        if guess_rho is None and self.rho is None:
            raise AttributeError('Must provide a guess density')
        else:
            rho = guess_rho
            self.old_rho = rho
        phi = np.sqrt(rho).ravel()
        res = minimize(fun=self.EnergyEvaluator,
                       jac=True,x0=phi,
                       method=self.optimization_method,
                       options=self.optimization_options)
        print(res.message)
        rho = DirectField(rho.grid,griddata_3d=np.reshape(res.x**2,np.shape(rho)),rank=1)
        self.rho = rho
        return rho



