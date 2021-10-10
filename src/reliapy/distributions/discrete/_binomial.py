from reliapy.distributions.discrete import _Discrete
from scipy.stats import binom as prob


class _Binomial(_Discrete):

    def __init__(self, k=None, n=None, p=None, loc=None, random_state=None):
        self.k = k
        self.n = n
        self.p = p
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
        return prob.pmf(X, k=self.k, n=self.n, p=self.p, loc=self.loc)

    def cdf(self, X=None):
        """
        CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            CDF of X.
        """
        return prob.cdf(X, k=self.k, n=self.n, p=self.p, loc=self.loc)

    def icdf(self, y=None):
        """
        Inverse CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            Inverse CDF of X.
        """
        return prob.ppf(y, n=self.n, p=self.p, loc=self.loc)

    def rvs(self, n_sim=1):
        """
        Get `n_sim` random samples.

        **Input:**
        * **n_sim** (`float`)
            Number of random samples.

        **Output**
            Samples.
        """
        return prob.rvs(n=self.n, p=self.p, loc=self.loc, size=n_sim, random_state=self.random_state)
