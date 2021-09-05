#from reliapy.math import phi_cdf
import numpy as np
from reliapy._messages import *
from reliapy.transformation._optimization import Optimization
from reliapy.math import *
#from reliapy.math import transform_xz, normal_equivalent


class FORM(Optimization):

    def __init__(self, limit_state_obj=None, distribution_obj=None, optimization='HLRF', decomposition='spectral'):
        self.distribution_obj = distribution_obj
        self.limit_state_obj = limit_state_obj
        self.optimization = optimization
        self.decomposition = decomposition
        self.pf = None
        self.beta = None

        super().__init__(limit_state_obj=limit_state_obj, distribution_obj=distribution_obj)

    def run(self, a=0.1, b=0.5, gamma=2, tol=1e-3, tol_1=1e-3, tol_2=1e-3, max_iter=20):

        if self.decomposition == 'spectral':
            Jyz, Jzy = spectral_decomposition(self.distribution_obj.correlation)
        elif self.decomposition == 'cholesky':
            Jyz, Jzy = cholesky_decomposition(self.distribution_obj.correlation)
        else:
            not_implemented_error()

        mean = self.distribution_obj.mean
        std = self.distribution_obj.std

        # M = mean
        #Jxy = np.diag(std)
        # Jyx = np.diag(np.reciprocal(std))

        nrv = self.distribution_obj.nrv
        x = mean
        itera = 0
        while itera < max_iter:

            Jxz, Jzx, M_eq, S_eq = transform_xz(x, distributions=self.distribution_obj)
            Jxy = Jxz @ Jzy
            Jyx = Jyz @ Jzx

            y = Jyx @ (x - M_eq)
            gx = self.limit_state_obj.function(x)
            dgdx = self.limit_state_obj.gradient(x)
            dgdy = Jxy.T @ dgdx
            alpha = dgdy / np.linalg.norm(dgdy)

            if self.optimization == 'iHLRF':
                y = self.iHLRF(a=0.1, b=0.5, gamma=2, tol=1e-3, tol_1=1e-3, tol_2=1e-3, max_iter=20)
            elif self.optimization == 'HLRF':
                y = self.HLRF(tol, max_iter)
            else:
                not_implemented_error()

            x = Jxy @ y + M_eq

            gy = self.limit_state_obj.function(x)
            dgdx = self.limit_state_obj.gradient(x)
            dgdy = Jxy.T @ dgdx

            error_1 = 1 + abs(np.dot(dgdy, y) / (np.linalg.norm(dgdy) * np.linalg.norm(y)))
            error_2 = np.linalg.norm(gy)

            if error_1 < tol_1 and error_2 < tol_2:
                break

            itera = itera + 1

        self.beta = np.linalg.norm(y)
        self.pf = phi_cdf(-self.beta)

