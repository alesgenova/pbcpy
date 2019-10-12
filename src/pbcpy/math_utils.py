import numpy as np
import scipy.special as sp
from scipy.optimize import minpack2
import time
from .constants import FFTLIB, MATHLIB, mathf
if FFTLIB == 'pyfftw' :
    import pyfftw

# Global variables
# FFT_SAVE = {
        # 'FFT_Grid' : np.zeros(3), 
        # 'IFFT_Grid' : np.zeros(3), 
        # 'FFT_OBJ' : None, 
        # 'IFFT_OBJ' : None,
        # 'RFFT_Grid' : np.zeros(3), 
        # 'RIFFT_Grid' : np.zeros(3), 
        # 'RFFT_OBJ' : None, 
        # 'RIFFT_OBJ' : None }
FFT_SAVE = {
        'FFT_Grid' : [np.zeros(3), np.zeros(3)],
        'IFFT_Grid' : [np.zeros(3), np.zeros(3)],
        'FFT_OBJ' : [None, None], 
        'IFFT_OBJ' : [None, None]}
# FFT_Grid = np.zeros(3)
# IFFT_Grid = np.zeros(3)
# FFT_OBJ = None
# IFFT_OBJ = None

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
TimeData = TimeObj()

def PYfft(grid, cplx = False, threads = 1):
    global FFT_SAVE
    if FFTLIB == 'pyfftw' :
        nr = grid.nr
        if np.all(nr == FFT_SAVE['FFT_Grid'][cplx]):
            fft_object = FFT_SAVE['FFT_OBJ'][cplx]
        else :
            if cplx :
                rA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
                cA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
            else :
                nrc = grid.nrG
                rA = pyfftw.empty_aligned(tuple(nr), dtype='float64')
                cA = pyfftw.empty_aligned(tuple(nrc), dtype='complex128')
            fft_object = pyfftw.FFTW(rA, cA, axes = (0, 1, 2),\
                    flags=('FFTW_MEASURE',), direction='FFTW_FORWARD', threads = threads)
            FFT_SAVE['FFT_Grid'][cplx] = nr
            FFT_SAVE['FFT_OBJ'][cplx] = fft_object
        return fft_object

def PYifft(grid, cplx = False, threads = 1):
    global FFT_SAVE
    if FFTLIB == 'pyfftw' :
        nr = grid.nrR
        if np.all(nr == FFT_SAVE['IFFT_Grid'][cplx]):
            fft_object = FFT_SAVE['IFFT_OBJ'][cplx]
        else :
            if cplx :
                rA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
                cA = pyfftw.empty_aligned(tuple(nr), dtype='complex128')
            else :
                nrc = grid.nr
                rA = pyfftw.empty_aligned(tuple(nr), dtype='float64')
                cA = pyfftw.empty_aligned(tuple(nrc), dtype='complex128')
                fft_object = pyfftw.FFTW(cA, rA, axes = (0, 1, 2), \
                        flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD', threads = threads)
            FFT_SAVE['IFFT_Grid'][cplx] = nr
            FFT_SAVE['IFFT_OBJ'][cplx] = fft_object
        return fft_object

def PowerInt(x, numerator, denominator = 1):
    y = x.copy()
    for i in range(numerator-1):
        np.multiply(y, x, out = y)
    if denominator == 1 :
        return y
    elif denominator == 2 :
        np.sqrt(y, out = y)
    elif denominator == 3 :
        np.cbrt(y, out = y)
    elif denominator == 4 :
        np.sqrt(y, out = y)
        np.sqrt(y, out = y)
    else :
        np.power(y, 1.0/denominator, out = y)
    return y

def bestFFTsize(N):
    '''
    http ://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs
    "FFTW is best at handling sizes of the form 2^a 3^b 5^c 7^d 11^e 13^f,  where e+f is either 0 or 1,  and the other exponents are arbitrary."
    '''
    a = int(np.log2(N)) + 2                                                                                                                      
    b = int(np.log(N)/np.log(3)) + 2                                                                                                             
    c = int(np.log(N)/np.log(5)) + 2                                                                                                             
    d = int(np.log(N)/np.log(7)) + 2                                                                                                             
    mgrid = np.mgrid[:a, :b, :c, :d].reshape(4, -1)                                                                                              
    arr0 = 2 ** mgrid[0] * 3 ** mgrid[1] * 5 ** mgrid[2] * 7 ** mgrid[3]
    arr1=arr0[np.logical_and(arr0>N/14, arr0<1.2*N)]
    arrAll=[]
    arrAll.extend(arr1)
    arrAll.extend(arr1*11)
    arrAll.extend(arr1*13)
    arrAll = np.asarray(arrAll)                                                                                                                  
    # bestN = np.min(arrAll[arrAll > N-1])   
    bestN = np.min(arrAll[arrAll > 0.99*N])   
    return bestN

def multiply(arr1, arr2):
    array1 = arr1
    array2 = arr2
    TP = False
    if MATHLIB == 'math_thran' :
        if arr1.flags['F_CONTIGUOUS'] and arr2.flags['F_CONTIGUOUS'] :
            TP = True
            array1 = arr1.T
            array2 = arr2.T
    elif MATHLIB == 'math_f2py' :
        if not arr1.flags['F_CONTIGUOUS'] and not arr2.flags['F_CONTIGUOUS'] :
            TP = True
            array1 = arr1.T
            array2 = arr2.T
    results = mathf.multiply(array1, array2)
    if TP :
        results = results.T
    return results

def add(arr1, arr2):
    array1 = arr1
    array2 = arr2
    TP = False
    if MATHLIB == 'math_thran' :
        if arr1.flags['F_CONTIGUOUS'] and arr2.flags['F_CONTIGUOUS'] :
            TP = True
            array1 = arr1.T
            array2 = arr2.T
    elif MATHLIB == 'math_f2py' :
        if not arr1.flags['F_CONTIGUOUS'] and not arr2.flags['F_CONTIGUOUS'] :
            TP = True
            array1 = arr1.T
            array2 = arr2.T
    results = mathf.add(array1, array2)
    if TP :
        results = results.T
    return results

def power(arr1, x):
    array1 = arr1
    TP = False
    if MATHLIB == 'math_thran' :
        if arr1.flags['F_CONTIGUOUS'] :
            TP = True
            array1 = arr1.T
    elif MATHLIB == 'math_f2py' :
        if not arr1.flags['F_CONTIGUOUS'] :
            TP = True
            array1 = arr1.T
    results = mathf.power(array1, x)
    if TP :
        results = results.T
    return results
