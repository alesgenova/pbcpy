from .field import DirectField, ReciprocalField
import numpy as np
import warnings

class Functional(object):
    '''
    Object representing a DFT functional
    
    Attributes
    ----------
    name: string
        The (optional) name of the functional

    energydensity: DirectField
        The energy density 

    potential: DirectField
        The first functional derivative of the functional wrt 
        the electron density 
        
    kernel: ReciprocalField
        The value of the reciprocal space kernel. This will
        be populated only if the functional is nonlocal
    '''


    def __init__(self, name='N/A', energydensity=None, potential=None, kernel=None):
        self.name = name
        if energydensity is not None:
            if isinstance(energydensity, DirectField):
                self.energydensity = energydensity
            else:
                print('Cazzarola!')
        if potential is not None:
            if isinstance(potential, DirectField):
                self.potential = potential
        if kernel is not None:
            if isinstance(kernel, (np.ndarray)):
                self.kernel = kernel


    def sum(self,other):
        energydensity = self.energydensity+other.energydensity
        potential = self.potential+other.potential
        return Functional(energydensity=energydensity,potential=potential)





