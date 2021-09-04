from reliapy.distributions.continuous import _Continuous
from scipy.stats import norm as prob


class Normal(_Continuous):

    def __init__(self, loc=None, scale=None, random_state=None):
        self.loc = loc
        self.scale = scale
        self.stats = prob.stats(loc=self.loc, scale=self.scale, moments='mv')
        self.random_state = random_state

        super().__init__(loc=loc, scale=scale)

    def pdf(self, X=None):
        return prob.pdf(X, loc=self.loc, scale=self.scale)

    def cdf(self, X=None):
        return prob.cdf(X, loc=self.loc, scale=self.scale)

    def icdf(self, y=None):
        return prob.ppf(y, loc=self.loc, scale=self.scale)

    def moment(self, n=1):
        return prob.moment(n, loc=self.loc, scale=self.scale)

    def rvs(self, n_sim=1):
        return prob.rvs(loc=self.loc, scale=self.scale, size=n_sim, random_state=self.random_state)
