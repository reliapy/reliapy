from reliapy.math import *


class Antithetic:
    """
    ``Antithetic`` simple random sampling.

     **Input:**
     * **distribution_obj** (`object`)
         Object of ``JointDistribution``.

    **Attributes:**
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

    def __init__(self, distribution_obj=None):

        self.distribution_obj = distribution_obj
        if not isinstance(distribution_obj.marginal, list):
            type_error('distributions', 'list')

        self.marginal = distribution_obj.marginal
        self.Cz = distribution_obj.Cz
        self.nrv = len(distribution_obj.marginal)
        self.random_state = distribution_obj.random_state
        self.decomposition = distribution_obj.decomposition

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
        Get random samples from the joint PDF.

        **Input:**
        * **n_sim** (`float`)
            Number of samples.

        **Output:**
        * **x** (`ndarray`)
            Random samples.

        """

        if self.decomposition == 'spectral':
            _, Jzy = spectral_decomposition(self.Cz)
        elif self.decomposition == 'cholesky':
            _, Jzy = cholesky_decomposition(self.Cz)
        else:
            not_implemented_error()

        # y = norm.rvs(loc=0, scale=1, size=(self.nrv, n_sim), random_state=self.random_state)
        # z = Jzy @ y

        U = np.random.rand(n_sim, self.nrv)
        U_ = 1 - U

        x0 = []
        x1 = []
        for i in range(n_sim):
            u = U[i, :]
            u_ = U_[i, :]

            xj0 = []
            xj0_ = []
            for j in range(self.nrv):
                x_ = phi_icdf(u[j])
                xj0.append(x_)

                x_ = phi_icdf(u_[j])
                xj0_.append(x_)

            zj = Jzy @ xj0
            zj_ = Jzy @ xj0_
            xj = []
            xj_ = []
            for j in range(self.nrv):
                p0 = phi_cdf(zj[j])
                x_ = self.marginal[j].icdf(p0)
                xj.append(x_)

                p0 = phi_cdf(zj_[j])
                x_ = self.marginal[j].icdf(p0)
                xj_.append(x_)

            x0.append(xj)
            x1.append(xj_)

        x0 = np.array(x0)
        x1 = np.array(x1)

        return x0, x1
