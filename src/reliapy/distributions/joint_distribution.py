from reliapy._messages import *
from scipy.stats._distn_infrastructure import rv_frozen
import numpy as np


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
