import numpy as np
import scipy.special as sp


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
