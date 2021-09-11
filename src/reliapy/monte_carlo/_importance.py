from reliapy.math import *
from reliapy.sampling import LHS, Antithetic, Random
from reliapy.transformation._optimization import Optimization
import copy


class Importance:
    """
    ``MonteCarlo`` is a class implementing the crude Monte Carlo simulation, and it is a child class of ``_Simulation``.

    **Input:**
    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    * **optimization** (`str`)
        Optimization method: `HLRF` or 'iHLRF' used in the searching of the design point.

    **Attributes:**

    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    * **n_sim** (`int`)
        Number of simulations.

    * **n_tasks** (`int`)
        Number of threads in parallel computing.

    * **random_state** (`int`)
        Random seed for the random number generator.

    * **pf** (`float`)
        Probability of failure.

    * **beta** (`float`)
        Reliability index.

    * **optimization** (`str`)
        Optimization method: `HLRF` or 'iHLRF' used in the searching of the design point.

    * **opt_obj** (`object`)
        Object of ``Optimization``

    """

    def __init__(self, limit_state_obj=None, sampling_obj=None, n_sim=None, n_tasks=1, random_state=None,
                 optimization='HLRF'):
        self.sampling_obj = sampling_obj
        self.n_sim = n_sim
        self.n_tasks = n_tasks
        self.limit_state_obj = limit_state_obj
        self.random_state = random_state
        self.pf = None
        self.beta = None
        self.optimization = optimization

        # set the random state of the ``JointDistribution`` object.
        self.sampling_obj.random_state = random_state

        self.opt_obj = Optimization(limit_state_obj=limit_state_obj,
                                    distribution_obj=self.sampling_obj.distribution_obj)

    def run(self, a=0.1, b=0.5, gamma=2, tol=1e-3, tol_1=1e-3, tol_2=1e-3, max_iter=20):
        """
        Run the Monte Carlo simulation with importance sampling centered in the design point.

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

        # Find the design point.
        if self.optimization == 'iHLRF':
            y_design = self.opt_obj.iHLRF(a=a, b=b, gamma=gamma, tol=tol, tol_1=tol_1, tol_2=tol_2, max_iter=max_iter)
        elif self.optimization == 'HLRF':
            y_design = self.opt_obj.HLRF(tol, max_iter)
        else:
            not_implemented_error()

        # Get the Jacobian for the transformation between Z and Y, and vice-versa.
        if self.sampling_obj.distribution_obj.decomposition == 'spectral':
            _, Jzy = spectral_decomposition(self.sampling_obj.distribution_obj.Cz)
        elif self.sampling_obj.distribution_obj.decomposition == 'cholesky':
            _, Jzy = cholesky_decomposition(self.sampling_obj.distribution_obj.Cz)
        else:
            not_implemented_error()

        # Transform the design point in Y to Z.
        z_design = Jzy @ y_design

        # Find the value of the design point in X.
        u = phi_cdf(z_design)
        x_design = []
        for j in range(self.sampling_obj.nrv):
            x_ = self.sampling_obj.marginal[j].icdf(u[j])
            x_design.append(x_)

        # Get `n_sim` random samples.
        if isinstance(self.sampling_obj, LHS) or isinstance(self.sampling_obj, Random):
            x_original = self.sampling_obj.rvs(n_sim=self.n_sim)
            x = copy.copy(x_original)
            for j in range(self.sampling_obj.nrv):
                x[:, j] = (x_original[:, j] - self.sampling_obj.mean[j]) + x_design[j]

            # Evaluate the limit state functions for the random samples.
            self.limit_state_obj.run(X=x)
            g = self.limit_state_obj.g

            num_failure = 0
            for i in range(self.n_sim):
                f = self.sampling_obj.distribution_obj.joint_pdf(x[i, :])
                h = self.sampling_obj.distribution_obj.joint_pdf(x_original[i, :])
                r = f/h
                if g[i] < 0:
                    num_failure = num_failure + r

            # Compute the probability of failure.
            self.pf = num_failure / self.n_sim

        else:

            x_original, x_original_ = self.sampling_obj.rvs(n_sim=self.n_sim)
            x = copy.copy(x_original)
            x_ = copy.copy(x_original_)
            for j in range(self.sampling_obj.nrv):
                x[:, j] = (x_original[:, j] - self.sampling_obj.mean[j]) + x_design[j]
                x_[:, j] = (x_original_[:, j] - self.sampling_obj.mean[j]) + x_design[j]

            # Evaluate the limit state functions for the random samples.
            self.limit_state_obj.run(X=x)
            g = self.limit_state_obj.g

            self.limit_state_obj.run(X=x_)
            g_ = self.limit_state_obj.g

            # Get the number of samples in the failure domain.
            num_failure = 0
            num_failure_ = 0
            for i in range(self.n_sim):
                f = self.sampling_obj.distribution_obj.joint_pdf(x[i, :])
                h = self.sampling_obj.distribution_obj.joint_pdf(x_original[i, :])
                r = f / h
                if g[i] < 0:
                    num_failure = num_failure + r

                f_ = self.sampling_obj.distribution_obj.joint_pdf(x_[i, :])
                h_ = self.sampling_obj.distribution_obj.joint_pdf(x_original_[i, :])
                r_ = f_ / h_
                if g_[i] < 0:
                    num_failure_ = num_failure_ + r_

            # Compute the probability of failure.
            pf = num_failure / self.n_sim
            pf_ = num_failure_ / self.n_sim

            # Compute the probability of failure.
            self.pf = (pf + pf_) / 2

        self.beta = pf2beta(self.pf)

