import numpy as np
from reliapy.math import numerical_gradient
# from multiprocessing import Pool


class LimitState:
    """
    ``LimitState`` is a class implementing the interface between the state limit function and reliapy.

    **Input:**
    * **limit_state_function** (`callable`)
        Limit state function.

    **Attributes:**

    * **g** (`float`)
        Result(s) of the limit state function.

    * **X** (`ndarray`)
        Random sample(s).

    * **n_sim** (`int`)
        Number of simulations.

    * **n_tasks** (`int`)
        Number of threads in parallel computing.

    """

    def __init__(self, limit_state_function=None, limit_state_gradient=None, n_tasks=1):

        if callable(limit_state_function):
            self.limit_state_function = limit_state_function
        else:
            raise TypeError('reliapy: `state_limit_function` must be callable.')

        if callable(limit_state_gradient):
            self.limit_state_gradient = limit_state_gradient
        else:
            self.state_limit_gradient = None

        self.X = None
        self.g = None
        self.n_sim = None
        self.n_tasks = n_tasks

    def function(self, X):
        """
        Get the values of the limit state function.

        **Input:**
        * **X** (`ndarray`)
            Samples of a random variable.

        """

        g = self.limit_state_function(X)

        return g

    def gradient(self, X):
        """
        Get the gradient of the limit state function either analytically or numerically.

        **Input:**
        * **X** (`ndarray`)
            Samples of a random variable.

        """

        if self.limit_state_gradient is None:
            # Get the gradient using finite differences.
            dg = numerical_gradient(X, self.limit_state_function)

        else:
            # Get the analytical gradient.
            dg = self.limit_state_gradient(X)

        return dg

    def run(self, X=None):
        """
        Get the responses of the limit state functions for the samples in `X`.

        **Input:**
        * **X** (`ndarray`)
            Samples of a random variable.

        """

        if X is None:
            raise ValueError('reliapy: `X` must be provided as input.')
        elif isinstance(X, list) or isinstance(X, np.ndarray):
            self.n_sim = len(X)  # This assumes that the number of rows is the number of simulations.
            self.X = X
        else:
            raise TypeError('reliapy: `X` must be passed either as a `list` or `ndarray`')

        if self.n_tasks == 1:
            self.g = self._run_serial(X)
        else:
            raise NotImplementedError('reliapy: multiprocessing not available yet.')

    def append(self, X=None):
        """
        Append responses of the limit state functions for the samples in `X`.

        **Input:**
        * **X** (`ndarray`)
            Samples to be appended of a random variable.

        """

        if X is None:
            raise ValueError('reliapy: `X` must be provided as input.')
        elif isinstance(X, list) or isinstance(X, np.ndarray):
            n_sim_append = len(X)  # This assumes that the number of rows is the number of simulations.
        else:
            raise TypeError('reliapy: `X` must be passed either as a `list` or `ndarray`')

        if self.n_tasks == 1:
            g_append = self._run_serial(X)

            for i in range(n_sim_append):
                self.g.append(g_append[i])
                self.X.append(X[i])
        else:
            raise NotImplementedError('reliapy: multiprocessing not available yet.')

        self.n_sim = self.n_sim + n_sim_append

    def _run_serial(self, X=None):
        """
        Private Method for performing the serial computation of the limit state function.

        **Input:**
        * **X** (`ndarray`)
            Samples of a random variable.

        **Output:**
        * **g** (`list`)
            Result(s) of the limit state function.

        """

        n_sim = len(X)  # This assumes that the number of rows is the number of simulations.

        # Run python model
        g = []
        for i in range(n_sim):
            state_lim = self.limit_state_function(X[i])
            g.append(state_lim)

        return g
