from reliapy._messages import *
from reliapy.math import *


class LHS:
    """
    ``LHS`` Latin Hypercube Sampling sampling.

    **Input:**
    * **distribution_obj** (`object`)
         Object of ``JointDistribution``.

    * **method** (`object`)
         Method to generate samples using LHS: `random` (conventional), and `center` (sample in the center).

    **Attributes:**
    * **method** (`object`)
         Method to generate samples using LHS: `random` (conventional), and `center` (sample in the center).

    * **distribution_obj** (`object`)
         Object of ``JointDistribution``.

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

    def __init__(self, distribution_obj=None, method='random'):

        self.distribution_obj = distribution_obj
        if not isinstance(distribution_obj.marginal, list):
            type_error('distributions', 'list')

        self.marginal = distribution_obj.marginal
        self.correlation = distribution_obj.correlation
        self.nrv = len(distribution_obj.marginal)
        self.random_state = distribution_obj.random_state
        self.decomposition = distribution_obj.decomposition
        self.method = method

        mean = []
        std = []
        for i in range(self.nrv):
            m = distribution_obj.marginal[i].stats[0]
            s = np.sqrt(distribution_obj.marginal[i].stats[1])
            mean.append(m)
            std.append(s)

        self.mean = np.array(mean)
        self.std = np.array(std)

    def rvs(self, n_sim=1):
        """
        Get random samples from the joint PDF using LHS.

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

        #y = norm.rvs(loc=0, scale=1, size=(self.nrv, n_sim), random_state=self.random_state)
        #z = Jzy @ y

        # Get a matrix of random permuations in each column.
        arr = np.arange(1, n_sim+1)
        P = []
        for i in range(self.nrv):
            P.append(np.random.permutation(arr))

        P = np.array(P).T

        # Get the matrix R, either random or full with 0.5.
        if self.method == 'random':
            R = np.random.rand(n_sim, self.nrv)
        elif self.method == 'center':
            R = np.ones((n_sim, self.nrv)) * 0.5
        else:
            not_implemented_error()

        S = (P - R) / n_sim

        x = []
        for i in range(n_sim):
            u = S[i, :]

            xj0 = []
            for j in range(self.nrv):
                x_ = phi_icdf(u[j])

                xj0.append(x_)

            zj = Jzy @ xj0
            xj = []
            for j in range(self.nrv):
                p0 = phi_cdf(zj[j])
                x_ = self.marginal[j].icdf(p0)
                xj.append(x_)

            x.append(xj)

        x = np.array(x)

        return x
