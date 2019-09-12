import numpy as np
from scipy.optimize import minimize, line_search
# from scipy.optimize.linesearch import line_search_wolfe1
from scipy.optimize.linesearch import scalar_search_wolfe1
from .field import DirectField
from .math_utils import LineSearchDcsrch,LineSearchDcsrch2

class LBFGS(object):

    def __init__(self, H0 = 1.0, Bound = 5):
        self.Bound = Bound
        self.H0 = H0
        self.s = []
        self.y = []
        self.rho = []

    def update(self, dx, dg):
        if len(self.s) > self.Bound :
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)
        self.s.append(dx)
        self.y.append(dg)
        rho = 1.0/np.einsum('ijkl->', dg * dx)
        self.rho.append(rho)

class Optimization(object):
    '''
    Class handling electron density optimization.
    minimizer based on scipy.minimize

    Attributes
    ---------
    optimization_method: string
            See scipy.minimize for available methods
            default: L-BFGS-B

    optimization_options: dict
            kwargs for the minim. method
            optional if method is L-BFGS-B

    EnergyEvaluator: TotalEnergyAndPotential class   
            

    guess_rho: DirectField, optional
            an initial guess for the electron density

     Example
     -------
     EE = TotalEnergyAndPotential(...)
     opt = Optimization(EnergyEvaluator=EE)
     new_rho = Optimization.get_optimal_rho(guess_rho)
    ''' 

    def __init__(self, 
                 optimization_method='CG-HS', 
                 optimization_options=None,
                 EnergyEvaluator=None,
                 guess_rho=None):
                    
        self.rho = guess_rho

        if optimization_options is None:
            self.optimization_options={}
            self.optimization_options["disp"] = None
            self.optimization_options["maxcor"] = 5
            self.optimization_options["ftol"] = 1.0e-7
            self.optimization_options["gtol"] = 1.0e-7
            self.optimization_options["maxfun"] = 1000
            self.optimization_options["maxiter"] = 100
            self.optimization_options["maxls"] = 10
            self.optimization_options["econv"] = 1E-5
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

    def get_direction(self, resA, dirA, phi=None, method='CG-HS', lbfgs=None, mu=None):

        if method[0:2] == 'CG' :
            if len(resA) == 1 :
                beta = 0.0
            elif method == 'CG-HS' and len(dirA) > 0 : #Maybe is the best of the CG.
                beta = np.einsum('ijkl->',resA[-1] *(resA[-1]-resA[-2]) ) / np.einsum('ijkl->',dirA[-1]*(resA[-1]-resA[-2]))
            elif  method == 'CG-FR':
                beta = np.einsum('ijkl->',resA[-1] ** 2) / np.einsum('ijkl->',resA[-2] ** 2) 
            elif method == 'CG-PR' :
                beta = np.einsum('ijkl->',resA[-1] *(resA[-1]-resA[-2]) ) / np.einsum('ijkl->',resA[-2] ** 2) 
            elif method == 'CG-DY' and len(dirA) > 0 :
                beta = np.einsum('ijkl->',resA[-1] **2 ) / np.einsum('ijkl->',dirA[-1]*(resA[-1]-resA[-2]))
            elif method == 'CG-CD' and len(dirA) > 0 :
                beta = -np.einsum('ijkl->',resA[-1] **2 ) / np.einsum('ijkl->',dirA[-1]*resA[-2])
            elif method == 'CG-LS' and len(dirA) > 0 :
                beta = np.einsum('ijkl->',resA[-1] *(resA[-1]-resA[-2]) ) / np.einsum('ijkl->',dirA[-1]*resA[-2])
            else :
                beta = np.einsum('ijkl->',resA[-1] ** 2) / np.einsum('ijkl->',resA[-2] ** 2) 

            if len(dirA) > 0 :
                direction = -resA[-1] + beta * dirA[-1]
            else :
                direction = -resA[-1]
            number = 1

        elif method == 'TN' :
            direction = np.zeros_like(resA[-1])
            epsi = 1.0E-7
            rho = phi ** 2
            if mu is None :
                func = self.EnergyEvaluator.ComputeEnergyPotential(rho, calcType = 'Potential')
                mu = (func.potential * rho).integral() / self.EnergyEvaluator.N
            res = -resA[-1]
            p = res
            r0Norm = np.einsum('ijkl->', res ** 2)
            r1Norm = r0Norm
            rConv = r0Norm * 0.1
            stat = 1
            #https ://en.wikipedia.org/wiki/Conjugate_gradient_method
            for it in range(50):
                phi1 = phi + epsi * p
                rho1 = phi1 ** 2
                func = self.EnergyEvaluator.ComputeEnergyPotential(rho1, calcType = 'Potential')
                Ap = ((func.potential * np.sign(phi1) - mu) * phi1 - resA[-1]) / epsi
                pAp = np.einsum('ijkl->', p * Ap)
                if pAp < 0.0 :
                    stat = 2
                    if it == 0 :
                        direction = res
                        stat = 3
                    print('!WARN : pAp small than zero :iter = ', it)
                    break
                alpha = r0Norm / pAp
                direction += alpha * p
                res -= alpha * Ap
                r1Norm = np.einsum('ijkl->', res ** 2)
                if r1Norm < rConv :
                    stat = 0  #convergence
                    break
                beta = r1Norm / r0Norm
                r0Norm = r1Norm
                p = res + beta * p 
            number = it

        elif method == 'LBFGS' :
            direction = np.zeros_like(resA[-1])
            rho = phi ** 2
            if mu is None :
                func = self.EnergyEvaluator.ComputeEnergyPotential(rho, calcType = 'Potential')
                mu = (func.potential * rho).integral() / self.EnergyEvaluator.N
            q = -resA[-1]
            alphaList = np.zeros(len(lbfgs.s))
            for i in range(len(lbfgs.s)-1, 0, -1):
                alpha = lbfgs.rho[i] * np.einsum('ijkl->', lbfgs.s[i] * q)
                alphaList[i] = alpha
                q -= alpha * lbfgs.y[i]

            if not lbfgs.H0 :
                if len(lbfgs.s) < 1 :
                    gamma = 1.0
                else :
                    gamma = np.einsum('ijkl->', lbfgs.s[-1] * lbfgs.y[-1]) / np.einsum('ijkl->', lbfgs.y[-1] * lbfgs.y[-1])
                direction = gamma * q
            else :
                direction = lbfgs.H0 * q

            for i in range(len(lbfgs.s)):
                beta = lbfgs.rho[i] * np.einsum('ijkl->', lbfgs.y[i] * direction)
                direction += lbfgs.s[i] * (alphaList[i]-beta)
            number = 1

        return direction, number


    def OrthogonalNormalization(self, p, phi):
        N = self.EnergyEvaluator.N
        p -= (p * phi).integral() / self.EnergyEvaluator.N * phi
        # print('Na', self.EnergyEvaluator.N, (phi * phi).integral())
        pNorm = (p ** 2).integral()
        # theta = np.sqrt( pNorm / self.EnergyEvaluator.N )
        theta = np.sqrt( pNorm / N)
        p *= np.sqrt(self.EnergyEvaluator.N / pNorm)
        return p, theta


    def optimize_rho(self, guess_rho = None):
        import time
        if guess_rho is None and self.rho is None:
            raise AttributeError('Must provide a guess density')
        else:
            rho = guess_rho
            self.old_rho = rho
        EnergyHistory = []

        BeginT = time.time()
        phi = np.sqrt(rho)
        func = self.EnergyEvaluator.ComputeEnergyPotential(rho)
        print('func time', time.time() - BeginT)
        mu = (func.potential* np.sign(phi)  * rho).integral() / self.EnergyEvaluator.N
        residual = (func.potential* np.sign(phi)  - mu)* phi
        residualA = []
        residualA.append(residual)
        theta = 0.5
        pk = -1
        directionA = []
        energy = func.energy
        EnergyHistory.append(energy)

        CostTime = time.time() - BeginT

        fmt = "{:8s}{:24s}{:16s}{:16s}{:8s}{:8s}{:16s}".format(\
                'Step','Energy(a.u.)', 'dE', 'dP', 'Nd', 'Nls', 'Time(s)')
        print(fmt)
        fmt = "{:<8d}".format(0)
        fmt += "{:<24.12E}".format(energy)
        fmt += "{:<16s}".format('+999999999E99')
        fmt += "{:<16.6E}".format(np.einsum('ijkl->', residual ** 2))
        fmt += "{:<8d}".format(1)
        fmt += "{:<8d}".format(1)
        fmt += "{:<16.6E}".format(CostTime)
        print(fmt)
        # exit(0)
        Bound = self.optimization_options["maxcor"]

        if self.optimization_method=='LBFGS' :
            # lbfgs = LBFGS(H0 = 1.0, Bound = Bound)
            lbfgs = LBFGS(H0 = None, Bound = Bound)

        for it in range(1, self.optimization_options["maxiter"]):
            if self.optimization_method=='LBFGS' :
                p, NumDirectrion = self.get_direction(residualA, directionA, phi=phi, method=self.optimization_method, lbfgs=lbfgs, mu=mu)
            else :
                p, NumDirectrion = self.get_direction(residualA, directionA, phi=phi, method=self.optimization_method, mu=mu)

            p, theta0 = self.OrthogonalNormalization(p, phi)

            def thetaEnergy(theta):
                newphi = phi * np.cos(theta) + p * np.sin(theta)
                f = self.EnergyEvaluator.ComputeEnergyPotential(newphi ** 2, calcType = 'Energy')
                # print('e111', f.energy, theta)
                return f.energy

            def thetaDerivative(theta):
                newphi = phi * np.cos(theta) + p * np.sin(theta)
                f = self.EnergyEvaluator.ComputeEnergyPotential(newphi ** 2, calcType = 'Potential')
                grad = np.sum(f.potential * np.sign(newphi) * newphi * (p * np.cos(theta) - phi * np.sin(theta)))
                # grad = np.sum(f.potential * newphi * (p * np.cos(theta) - phi * np.sin(theta)))
                return grad * 2.0

            def EnergyAndDerivative(theta):
                newphi = phi * np.cos(theta) + p * np.sin(theta)
                f = self.EnergyEvaluator.ComputeEnergyPotential(newphi ** 2, calcType = 'Both')
                grad = np.sum(f.potential * np.sign(newphi) * newphi * (p * np.cos(theta) - phi * np.sin(theta)))
                return [f.energy, grad * 2.0]

            if thetaDerivative(0.0) > 0 :
                p = -residualA[-1]
                p, theta0 = self.OrthogonalNormalization(p, phi)
                print('!WARN: Change to steepest decent')
                # print('theta0', theta0)

            # theta = line_search(thetaEnergy, thetaDerivative, min(theta0, theta), pk, c1 = 1e-4, c2 = 0.2)[0]
            # theta = line_search(thetaEnergy, thetaDerivative, 0.2, pk, c1 = 1e-4, c2 = 0.2)[0]
            # theta = line_search_wolfe1(thetaEnergy, thetaDerivative, 0.2, pk, c1 = 1e-4, c2 = 0.2)[0]
            # theta = scalar_search_wolfe1(thetaEnergy, thetaDerivative, min(theta0, theta), pk, c1 = 1e-4, c2 = 0.2)[0]
            # theta = scalar_search_wolfe1(thetaEnergy, thetaDerivative, min(theta0, theta), zero, thetaDerivative(zero), pk, c1 = 1e-4, c2 = 0.2)[0]
            # theta = scalar_search_wolfe1(thetaEnergy, thetaDerivative, phi0=0.2, old_phi0=0.0, derphi0 = 1.0, c1 = 1e-4, c2 = 0.2, amin=1e-6, xtol=1e-10)[0]
            # theta = scalar_search_wolfe1(thetaEnergy, thetaDerivative, c1 = 1e-4, c2 = 0.2, amin=1e-6, xtol=1e-10)[0]
            # phi = phi * np.cos(theta) + p * np.sin(theta)
            # func = self.EnergyEvaluator.ComputeEnergyPotential(phi ** 2)

            theta = min(theta0, theta)
            # print('thetaIni', theta, theta0)
            # theta,_, _, task, NumLineSearch =  LineSearchDcsrch(thetaEnergy, thetaDerivative, alpha0 = theta, func0=energy, 
                    # derfunc0=thetaDerivative(0.0), c1=1e-4, c2=0.1, amax=np.pi, amin=0.0, xtol=1e-12, maxiter = 100)

            func0 = EnergyAndDerivative(0.0)
            theta,_, _, task, NumLineSearch =  LineSearchDcsrch2(EnergyAndDerivative, alpha0 = theta,
                   func0 = func0, c1=1e-4, c2=0.1, amax=np.pi, amin=0.0, xtol=1e-12, maxiter = 100)

            # print('final theta', theta)
            ### Just for LBFGS 
            old_phi = phi
            phi = phi * np.cos(theta) + p * np.sin(theta)
            rho = phi ** 2
            func = self.EnergyEvaluator.ComputeEnergyPotential(rho)
            mu = (func.potential* np.sign(phi)  * rho).integral() / self.EnergyEvaluator.N
            residual = (func.potential* np.sign(phi)  - mu)* phi
            residualA.append(residual)

            if self.optimization_method=='LBFGS' :
                lbfgs.update(phi-old_phi, residualA[-1]-residualA[-2])

            energy = func.energy
            EnergyHistory.append(energy)
            CostTime = time.time() - BeginT
            #
            fmt = "{:<8d}".format(it)
            fmt += "{:<24.12E}".format(energy)
            fmt += "{:<16.6E}".format(EnergyHistory[-1]-EnergyHistory[-2])
            fmt += "{:<16.6E}".format(np.einsum('ijkl->', residual ** 2))
            fmt += "{:<8d}".format(NumDirectrion)
            fmt += "{:<8d}".format(NumLineSearch)
            fmt += "{:<16.6E}".format(CostTime)
            print(fmt)
            if abs(EnergyHistory[-1]-EnergyHistory[-2]) < self.optimization_options["econv"] :
                print('#### Density Optimization Converged ####')
                break

            directionA.append(p)
            if len(residualA) > 2 :
                residualA.pop(0)
            if len(directionA) > 2 :
                directionA.pop(0)

        return phi ** 2
