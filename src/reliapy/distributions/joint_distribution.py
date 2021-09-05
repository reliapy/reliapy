from reliapy._messages import *
from scipy.stats import norm
import numpy as np
from reliapy.math import spectral_decomposition, cholesky_decomposition


class JointDistribution:
    """
    ``JointDistribution`` is a class implementing the elements of a joint distribution that are useful for reliability.

     **Input:**
    * **marginal** (`list`)
        A list of objects of marginal distribution.

    * **correlation** (`ndarray`)
        Correlation matrix.

    * **random_state** (`float`, `int`)
        Random seed.

    * **decomposition** (`str`)
        Decomposition of the correlation method: `spectral` or `cholesky`.

    **Attributes:**

    * **marginal** (`list`)
        A list of objects of marginal distribution.

    * **correlation** (`ndarray`)
        Correlation matrix.

    * **nrv** (`int`)
        Number of random variables.

    * **random_state** (`float`, `int`)
        Random seed.

    * **decomposition** (`str`)
        Decomposition of the correlation method: `spectral` or `cholesky`.

    * **mean** (`ndarray`)
        Array of means.

    * **std** (`ndarray`)
        Array of standard deviations.

    """

    def __init__(self, marginal=None, correlation=None, random_state=None, decomposition='spectral'):

        if not isinstance(marginal, list):
            type_error('distributions', 'list')

        self.marginal = marginal
        self.correlation = correlation
        self.nrv = len(marginal)
        self.random_state = random_state
        self.decomposition = decomposition

        mean = []
        std = []
        for i in range(self.nrv):
            m = marginal[i].stats[0]
            s = np.sqrt(marginal[i].stats[1])
            mean.append(m)
            std.append(s)

        self.mean = np.array(mean)
        self.std = np.array(std)

    def rvs(self, n_sim=1):
        """
        Get random samples from the joint PDF.

        **Input:**
        * **n_sim** (`float`)
            Number of samples.

        **Output:**
        * **x** (`ndarray`)
            Random samples.

        """

        if self.decomposition == 'spectral':
            _, Jzy = spectral_decomposition(self.correlation)
        elif self.decomposition == 'cholesky':
            _, Jzy = cholesky_decomposition(self.correlation)
        else:
            not_implemented_error()

        y = norm.rvs(loc=0, scale=1, size=(self.nrv, n_sim), random_state=self.random_state)
        z = Jzy @ y

        x = []
        for i in range(n_sim):
            u = norm.cdf(z[:, i], loc=0, scale=1)

            xj = []
            for j in range(self.nrv):
                x_ = self.marginal[j].icdf(u[j])
                xj.append(x_)

            x.append(xj)

        x = np.array(x)

        return x
