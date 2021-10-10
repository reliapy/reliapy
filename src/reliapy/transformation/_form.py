from reliapy.transformation._optimization import Optimization
from reliapy.math import *


class FORM(Optimization):
    """
    ``FORM`` is a class implementing the First Order Reliability Method (FORM).

     **Input:**
    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    * **optimization** (`str`)
        Optimization method: `HLRF` or 'iHLRF'.

    * **decomposition** (`str`)
        Decomposition of the correlation method: `spectral` or `cholesky`.

    **Attributes:**

    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    * **optimization** (`str`)
        Optimization method: `HLRF` or 'iHLRF'.

    * **decomposition** (`str`)
        Decomposition of the correlation method: `spectral` or `cholesky`.

    * **pf** (`float`)
        Probability of failure.

    * **beta** (`float`)
        Reliability index.

    """

    def __init__(self, limit_state_obj=None, distribution_obj=None, optimization='HLRF', decomposition='spectral'):
        self.distribution_obj = distribution_obj
        self.limit_state_obj = limit_state_obj
        self.optimization = optimization
        self.decomposition = decomposition
        self.pf = None
        self.beta = None
        self.design_point_x = None
        self.design_point_y = None

        super().__init__(limit_state_obj=limit_state_obj, distribution_obj=distribution_obj)

    def run(self, a=0.1, b=0.5, gamma=2, tol=1e-3, tol_1=1e-3, tol_2=1e-3, max_iter=20):
        """
        Run FORM.

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

        # Get the Jacobians between Y and Z (and vice-versa).
        if self.decomposition == 'spectral':
            Jyz, Jzy = spectral_decomposition(self.distribution_obj.Cz)
        elif self.decomposition == 'cholesky':
            Jyz, Jzy = cholesky_decomposition(self.distribution_obj.Cz)
        else:
            not_implemented_error()

        # Get the array with the means of the random variables.
        mean = self.distribution_obj.mean
        std = self.distribution_obj.std

        # Check if it is a system or not.
        g_mean = self.limit_state_obj.function(mean)
        if isinstance(g_mean, tuple):
            n_lse = len(g_mean) - 1
        elif isinstance(g_mean, float):
            n_lse = 1
        else:
            not_implemented_error()

        # Start the iteration guessing the design point.
        if n_lse == 1:
            y, x = self._iteration_form(mean, std, a, b, gamma, tol, tol_1, tol_2, max_iter, Jzy, Jyz,
                                        sys=False, sys_id=None)

            beta = np.linalg.norm(y)
            pf = beta2pf(beta)
            design_point_y = y
            design_point_x = x

        else:

            beta = []
            pf = []
            design_point_y = []
            design_point_x = []
            for k in range(n_lse):
                y, x = self._iteration_form(mean, std, a, b, gamma, tol, tol_1, tol_2, max_iter, Jzy, Jyz,
                                            sys=True, sys_id=k)

                beta.append(np.linalg.norm(y))
                pf.append(beta2pf(np.linalg.norm(y)))
                design_point_y.append(y)
                design_point_x.append(x)

        self.beta = beta
        self.pf = pf
        self.design_point_y = design_point_y
        self.design_point_x = design_point_x

    def _iteration_form(self, mean, std, a, b, gamma, tol, tol_1, tol_2, max_iter, Jzy, Jyz, sys, sys_id):

        # nrv = self.distribution_obj.nrv
        x = mean
        g0 = self.limit_state_obj.function(x)

        if sys:
            g0 = g0[sys_id + 1]

        tol_ = tol * np.linalg.norm(g0)
        itera = 0
        while itera < max_iter:
            # Get the jacobians between X and Y (and vice versa) using the composition scheme.
            Jxz, Jzx, M_eq, S_eq = transform_xz(x, distributions=self.distribution_obj)
            Jxy = Jxz @ Jzy
            Jyx = Jyz @ Jzx

            # Transform the point x from X to Y
            y = Jyx @ (x - M_eq)

            # Get the sensitive indexes. (Not currently used. This can change in the future)
            if sys:
                gy = self.limit_state_obj.function(x)
                gy = gy[sys_id + 1]
                dgdx = self.limit_state_obj.gradient(x)
                dgdx = dgdx[sys_id]

            else:
                gy = self.limit_state_obj.function(x)
                dgdx = self.limit_state_obj.gradient(x)

            dgdy = Jxy.T @ dgdx
            # alpha = dgdy / np.linalg.norm(dgdy)

            # Update y.
            if self.optimization == 'iHLRF':
                # y = self.iHLRF(a=a, b=b, gamma=gamma, tol=tol, tol_1=tol_1, tol_2=tol_2, max_iter=max_iter,
                #               sys_id=sys_id)

                y = self.update_iHLRF(y, gy, dgdy, gamma, a, b, mean, std, sys, sys_id, tol_)

            elif self.optimization == 'HLRF':
                # y = self.HLRF(tol, max_iter, sys_id)
                y = self.update_HLRF(y, gy, dgdy)

            else:
                not_implemented_error()

            #c = (np.dot(dgdy, y) - gy) / (np.linalg.norm(dgdy) ** 2)
            #y = c * dgdy

            # Transform y from Y to X.
            x = Jxy @ y + M_eq

            # Evaluate g(y) and its gradient.
            if sys:
                gy_ = self.limit_state_obj.function(x)
                dgdx_ = self.limit_state_obj.gradient(x)
                gy = gy_[sys_id + 1]
                dgdx = dgdx_[sys_id]

            else:
                gy = self.limit_state_obj.function(x)
                dgdx = self.limit_state_obj.gradient(x)

            # gy = self.limit_state_obj.function(x)
            # dgdx = self.limit_state_obj.gradient(x)

            dgdy = Jxy.T @ dgdx

            # Compute the errors.
            error_1 = 1 + abs(np.dot(dgdy, y) / (np.linalg.norm(dgdy) * np.linalg.norm(y)))
            error_2 = np.linalg.norm(gy)

            if error_1 < tol_1 and error_2 < tol_2:
                break

            itera = itera + 1

        return y, x

