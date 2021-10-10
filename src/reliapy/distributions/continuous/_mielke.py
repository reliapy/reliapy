from reliapy.distributions.continuous import _Continuous
from scipy.stats import mielke as prob


class Mielke(_Continuous):

    def __init__(self, k=None, s=None, loc=None, scale=None, random_state=None):
        self.k = k
        self.s = s
        self.loc = loc
        self.scale = scale
        self.stats = prob.stats(a=self.k, b=self.s, loc=self.loc, scale=self.scale, moments='mv')
        self.random_state = random_state

        super().__init__()

    def pdf(self, X=None):
        """
        PDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            PDF of X.
        """
        return prob.pdf(X, k=self.k, s=self.s, loc=self.loc, scale=self.scale)

    def cdf(self, X=None):
        """
        CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            CDF of X.
        """
        return prob.cdf(X, k=self.k, s=self.s, loc=self.loc, scale=self.scale)

    def icdf(self, y=None):
        """
        Inverse CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            Inverse CDF of X.
        """
        return prob.ppf(y, k=self.k, s=self.s, loc=self.loc, scale=self.scale)

    def moment(self, n=1):
        """
        Get the non-central moments of order n.

        **Input:**
        * **n** (`float`)
            Order of the moment.

        **Output**
            non central moment.
        """
        return prob.moment(n, k=self.k, s=self.s, loc=self.loc, scale=self.scale)

    def rvs(self, n_sim=1):
        """
        Get `n_sim` random samples.

        **Input:**
        * **n_sim** (`float`)
            Number of random samples.

        **Output**
            Samples.
        """
        return prob.rvs(k=self.k, s=self.s, loc=self.loc, scale=self.scale, size=n_sim, random_state=self.random_state)
