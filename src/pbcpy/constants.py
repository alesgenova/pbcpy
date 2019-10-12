
### Import fft library
try:
    import pyfftw
    FFTLIB = 'pyfftw'
except :
    FFTLIB = 'numpy'
# FFTLIB = 'numpy'

### Import math library
# try:
    # from .src_f.math_f import math_f as mathf
    # MATHLIB = 'math_f2py'
# except Exception as e:
    # try:
        # from . import math_thran as mathf
        # MATHLIB = 'math_thran'
    # except Exception as e:
        # import numpy as mathf
        # MATHLIB = 'numpy'
# from .src_f.math_f import math_f as mathf
# MATHLIB = 'math_f2py'
# from . import math_thran as mathf
# MATHLIB = 'math_thran'
import numpy as mathf
MATHLIB = 'numpy'

print('Use "%s" for Fourier Transform' %(FFTLIB))
print('Use "%s" for some mathematical calculations' %(MATHLIB))

LEN_UNITS = ['Bohr', 'Angstrom', 'nm', 'm']

LEN_CONV = {}
LEN_CONV['Bohr'] = {
    'Bohr': 1.0, 'Angstrom': 0.5291772106712,
    'nm': 0.05291772106712, 'm': 5.291772106712e-11
}
LEN_CONV['Angstrom'] = {
    'Bohr': 1.8897261254535427,
    'Angstrom': 1.0, 'nm': 1.0e-1, 'm': 1.0e-10
}
LEN_CONV['nm'] = {
    'Bohr': 18.897261254535427,
    'Angstrom': 10., 'nm': 1.0, 'm': 1.0e-9
}
LEN_CONV['m'] = {
    'Bohr': 1.8897261254535427e10,
    'Angstrom': 1.0e10, 'nm': 1.0e9, 'm': 1.0
}

ENERGY_CONV = {}
ENERGY_CONV['eV'] = {
        'eV': 1.0, 'Hartree' : 0.03674932598150397
}
ENERGY_CONV['Hartree'] = {
        'eV' : 27.2113834279111, 'Hartree': 1.0
}

units_warning = "Please only feed pbcpy quantities in atomic units (Bohr). An automatic units system might be implemented in the future"
