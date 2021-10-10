from reliapy.distributions.discrete import _Discrete
from scipy.stats import poisson as prob


class _Poisson(_Discrete):

    def __init__(self, k=None, mu=None, loc=None, random_state=None):
        self.k = k
        self.mu = mu
        self.loc = loc
        self.stats = prob.stats(p=self.p, loc=self.loc, moments='mv')
        self.random_state = random_state

        super().__init__()

    def pmf(self, X=None):
        """
        PMF (probability mass function).

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            PDF of X.
        """
        return prob.pmf(X, k=self.k, mu=self.mu, loc=self.loc)

    def cdf(self, X=None):
        """
        CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            CDF of X.
        """
        return prob.cdf(X, k=self.k, mu=self.mu, loc=self.loc)

    def icdf(self, y=None):
        """
        Inverse CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            Inverse CDF of X.
        """
        return prob.ppf(y, mu=self.mu, loc=self.loc)

    def rvs(self, n_sim=1):
        """
        Get `n_sim` random samples.

        **Input:**
        * **n_sim** (`float`)
            Number of random samples.

        **Output**
            Samples.
        """
        return prob.rvs(mu=self.mu, loc=self.loc, size=n_sim, random_state=self.random_state)
