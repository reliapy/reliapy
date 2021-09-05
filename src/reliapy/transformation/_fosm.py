from reliapy.math import *
import numpy as np
from reliapy._messages import *
from reliapy.transformation._optimization import Optimization


class FOSM(Optimization):
    """
    ``FOSM`` is a class implementing the First Order Second Moment method (FOSM).

     **Input:**
    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    * **optimization** (`str`)
        Optimization method: `HLRF` or 'iHLRF'.

    **Attributes:**

    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    * **optimization** (`str`)
        Optimization method: `HLRF` or 'iHLRF'.

    * **pf** (`float`)
        Probability of failure.

    * **beta** (`float`)
        Reliability index.

    """

    def __init__(self, limit_state_obj=None, distribution_obj=None, optimization='HLRF'):
        self.distribution_obj = distribution_obj
        self.limit_state_obj = limit_state_obj
        self.optimization = optimization
        self.pf = None
        self.beta = None

        super().__init__(limit_state_obj=limit_state_obj, distribution_obj=distribution_obj)

    def run(self, a=0.1, b=0.5, gamma=2, tol=1e-3, tol_1=1e-3, tol_2=1e-3, max_iter=20):
        """
        Run FOSM.

        **Input:**
        * **a** (`float`)
            Parameter `a` must be a value in the interval (0, 1).

        * **b** (`float`)
            Parameter `b` must be a value in the interval (0, 1).

        * **gamma** (`float`)
            Parameter `gamma` must be a value larger than or equal to 1.

        * **tol_1** (`float`)
            Error tolerance for a given criteria.

        * **tol_2** (`float`)
            Error tolerance for a given criteria.

        * **tol** (`float`)
            Error tolerance for convergence.

        * **max_iter** (`float`)
            Maximum number of iterations.

        """

        # Get the mean and standard deviation of the random variables.
        mean = self.distribution_obj.mean
        std = self.distribution_obj.std

        # Get the jacobian for the transformation between X and Y.
        Jxy = np.diag(std)

        # Start the iterative problem setting x equal to the mean of the random variables.
        x = mean
        itera = 0
        while itera < max_iter:

            # Evaluate the limit state function and its gradient in X.
            # gx = self.limit_state_obj.function(x)  # todo: use of gx.
            dgdx = self.limit_state_obj.gradient(x)

            # Transform the point from X to Y
            y = (x - mean) / std

            # Transform the gradient from X to Y.
            # dgdy = Jxy.T @ dgdx  # todo: used to estimate alpha.

            # Get the sensitivity indexes.
            # alpha = dgdy / np.linalg.norm(dgdy)  # todo: use of the alpha for eliminating some variables.

            # Get the next point y.
            if self.optimization == 'iHLRF':
                y = self.iHLRF(a=a, b=b, gamma=gamma, tol=tol, tol_1=tol_1, tol_2=tol_2, max_iter=max_iter)
            elif self.optimization == 'HLRF':
                y = self.HLRF(tol, max_iter)
            else:
                not_implemented_error()

            # Transform y from Y to X.
            x = Jxy @ y + mean

            # Evaluate g(y), dg/dx and dg/dy.
            gy = self.limit_state_obj.function(x)
            dgdx = self.limit_state_obj.gradient(x)
            dgdy = Jxy.T @ dgdx

            # Check errors.
            error_1 = 1 + abs(np.dot(dgdy, y) / (np.linalg.norm(dgdy) * np.linalg.norm(y)))
            error_2 = np.linalg.norm(gy)

            if error_1 < tol_1 and error_2 < tol_2:
                break

            itera = itera + 1
        
        self.beta = np.linalg.norm(y)
        self.pf = beta2pf(self.beta)

