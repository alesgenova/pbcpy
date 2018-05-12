import numpy as np
class System(object):

    def __init__(self, ions, cell, name=None, field=None):

        self.ions = ions
        self.cell = cell
        self.name = name
        self.field = field
        self.natoms = np.shape(ions)[0]
