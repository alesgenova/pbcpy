from .grid_functions import Grid_Function_Base, Grid_Function, Grid_Function_Reciprocal, Grid_Space

class Functional(object):
	'''
	Object representing a DFT functional
	
	Attributes
        ----------
        name: string
		The (optional) name of the functional

        energydensity: Grid_Function
		The energy density 

	potential: Grid_Function
		The first functional derivative of the functional wrt 
		the electron density 
        
        kernel: Grid_Function_Reciprocal
		The value of the reciprocal space kernel. This will
		be populated only if the functional is nonlocal
	'''


    def __init__(self, name=None, energydensity=None, potential=None, kernel=None):
        if energydensity is not None:
        	if isinstance(energydensity, Grid_Function):
            		self.energydensity = energydensity
        if potential is not None:
        	if isinstance(potential, Grid_Function):
            		self.potential = potential
        if name is not None:
            if isinstance(name,basestring):
            	self.name = name
        if kernel is not None:
            if isinstance(g, Grid_Function_Reciprocal):
            	self.kernel = kernel



