import numpy as np
import scipy.special as sp
from scipy.optimize import minpack2
from .constants import FFTLIB
import time

# Global variables
FFT_Grid = np.zeros(3)
IFFT_Grid = np.zeros(3)
FFT_OBJ = None
IFFT_OBJ = None

class CBspline(object):
    '''
    the Cardinal B-splines
    '''
    def __init__(self, ions = None, rho = None, order = 10, **kwargs):
        self._order = order
        self._Mn = None

        if ions is not None:
            self.ions      = ions
        else:
            raise AttributeError("Must pass ions to CBspline")

        if rho is not None:
            self.rho       = rho
        else:
            raise AttributeError("Must pass rho to CBspline")

    @property
    def order(self):
        return self._order

    @property
    def bm(self):
        return self._calc_bm()

    def calc_Mn(self, x, order = None):
        '''
        x -> [0, 1)
        x --> u + {0, 1, ..., order}
        u -> [0, 1), [1, 2),...,[order, order + 1)
        M_n(u) = u/(n-1)*M_(n-1)(u) + (n-u)/(n-1)*M_(n-1)(u-1)
        '''
        if not order :
            order = self.order

        Mn = np.zeros(self.order + 1)
        Mn[1] = x
        Mn[2] = 1.0 - x
        for i in range(3, order + 1):
            for j in range(0, i):
                n = i - j
                # Mn[n] = (x + n - 1) * Mn[n] + (i - (x + n - 1)) * Mn[n - 1]
                Mn[n] = (x + n - 1) * Mn[n] + (j + 1 - x) * Mn[n - 1]
                Mn[n] /= (i  - 1)
        return Mn

    def _calc_bm(self):
        nr = self.rho.grid.nr
        Mn = self.calc_Mn(1.0)
        bm = []
        for i in range(3):
            q = 2.0 * np.pi * np.arange(nr[i]) / nr[i]
            tmp = np.exp(-1j * (self.order - 1.0) * q)
            bm.append(tmp)
            factor = np.zeros_like(bm[i])
            for k in range(1, self.order):
                factor += Mn[k] * np.exp(-1j * k * q)
            bm[i] /= factor
        return bm

def LineSearchDcsrch(func, derfunc, alpha0 = None, func0=None, derfunc0=None,
        c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter = 100):

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    if alpha0 is None :
        alpha0 = 0.0
        func0 = func(alpha0)
        derfunc0 = derfunc(alpha0)

    alpha1 = alpha0
    func1 = func0
    derfunc1 = derfunc0

    for i in range(maxiter):
        alpha1, func1, derfunc1, task = minpack2.dcsrch(alpha1, func1, derfunc1,
                                                   c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        if task[:2] == b'FG':
            func1 = func(alpha1)
            derfunc1 = derfunc(alpha1)
        else:
            break
    else:
        alpha1 = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        alpha1 = None  # failed

    return alpha1, func1, derfunc1, task, i

def LineSearchDcsrch2(func,alpha0 = None, func0=None, \
        c1=1e-4, c2=0.9, amax=1.0, amin=0.0, xtol=1e-14, maxiter = 100):

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    if alpha0 is None :
        alpha0 = 0.0
        func0 = func(alpha0)

    alpha1 = alpha0
    x1 = func0[0]
    g1 = func0[1]

    for i in range(maxiter):
        alpha1, x1, g1, task = minpack2.dcsrch(alpha1, x1, g1, c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        if task[:2] == b'FG':
            func1 = func(alpha1)
            x1 = func1[0]
            g1 = func1[1]
        else:
            break
    else:
        alpha1 = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        alpha1 = None  # failed

    return alpha1, x1, g1, task, i

class TimeObj(object):
    '''
    '''
    def __init__(self,  **kwargs):
        self.labels = []
        self.tic = {}
        self.toc = {}
        self.cost = {}
        self.number = {}

    def Begin(self, label):
        if label in self.tic :
            self.number[label] += 1
        else :
            self.labels.append(label)
            self.number[label] = 1
            self.cost[label] = 0.0

        self.tic[label] = time.time()

    def End(self, label):
        if label not in self.tic :
            print(' !!! ERROR : You should add "Begin" before this')
        else :
            self.toc[label] = time.time()
            t = time.time() - self.tic[label]
            self.cost[label] += t
        return t

def PYfft(grid):
    global FFT_Grid, FFT_OBJ
    if FFTLIB == 'pyfftw' :
        import pyfftw
        nr = grid.nr
        if np.all(nr == FFT_Grid): 
            fft_object = FFT_OBJ
        else :
            nrc = nr.copy()
            nrc[-1]= nrc[-1]//2 + 1
            rA = pyfftw.empty_aligned(tuple(nr), dtype='float64')
            cA = pyfftw.empty_aligned(tuple(nrc), dtype='complex128')
            # print ('Threads:' , multiprocessing.cpu_count())
            fft_object = pyfftw.FFTW(rA, cA, axes = (0, 1, 2), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD')
            # fft_object = pyfftw.FFTW(rA, cA, axes = (0, 1, 2), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD',threads=4)
            FFT_OBJ = fft_object
            FFT_Grid = nr
        return fft_object

def PYifft(grid):
    global IFFT_Grid, IFFT_OBJ
    if FFTLIB == 'pyfftw' :
        import pyfftw
        nr = grid.nr
        if np.all(nr == IFFT_Grid): 
            fft_object = IFFT_OBJ
        else :
            nr = grid.nr
            nrc = nr.copy()
            nrc[-1]= nrc[-1]//2 + 1
            rA = pyfftw.empty_aligned(tuple(nr), dtype='float64')
            cA = pyfftw.empty_aligned(tuple(nrc), dtype='complex128')
            fft_object = pyfftw.FFTW(cA, rA, axes = (0, 1, 2), flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD')
            IFFT_OBJ = fft_object
            IFFT_Grid= nr
        return fft_object

def PowerInt(x, numerator, denominator = 1):
    y = x.copy()
    for i in range(numerator - 1):
        y *= x
    if denominator == 2 :
        y = np.sqrt(y)
    elif denominator == 3 :
        y = np.cbrt(y)
    elif denominator == 4 :
        y = np.sqrt(np.sqrt(y))
    else :
        y = y ** (1.0/denominator)
    return y

TimeData = TimeObj()
