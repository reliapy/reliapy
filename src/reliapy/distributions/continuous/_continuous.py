import scipy as sp
import numpy as np
from reliapy._messages import *


class _Continuous:

    def __init__(self, loc=None, scale=None, random_state=None):
        self.loc = loc
        self.scale = scale
        self.central_moments = None
        self.random_state = random_state

    def pdf(self, X=None):
        template_error()

    def cdf(self, X=None):
        template_error()

    def moment(self, n=1):
        template_error()

    def rvs(self, n_sim=1):
        template_error()
