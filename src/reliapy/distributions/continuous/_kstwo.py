from reliapy.distributions.continuous import _Continuous
from scipy.stats import kstwo as prob


class KStwo(_Continuous):

    def __init__(self, n=None, loc=None, scale=None, random_state=None):
        self.n = n
        self.loc = loc
        self.scale = scale
        self.stats = prob.stats(c=self.c, loc=self.loc, scale=self.scale, moments='mv')
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
        return prob.pdf(X, n=self.n, loc=self.loc, scale=self.scale)

    def cdf(self, X=None):
        """
        CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            CDF of X.
        """
        return prob.cdf(X, n=self.n, loc=self.loc, scale=self.scale)

    def icdf(self, y=None):
        """
        Inverse CDF.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
            Inverse CDF of X.
        """
        return prob.ppf(y, n=self.n, loc=self.loc, scale=self.scale)

    def moment(self, n=1):
        """
        Get the non-central moments of order n.

        **Input:**
        * **n** (`float`)
            Order of the moment.

        **Output**
            non central moment.
        """
        return prob.moment(n, n=self.n, loc=self.loc, scale=self.scale)

    def rvs(self, n_sim=1):
        """
        Get `n_sim` random samples.

        **Input:**
        * **n_sim** (`float`)
            Number of random samples.

        **Output**
            Samples.
        """
        return prob.rvs(n=self.n, loc=self.loc, scale=self.scale, size=n_sim, random_state=self.random_state)
