import scipy as sp
import numpy as np
from reliapy._messages import *
from reliapy.monte_carlo._simulation import _Simulation


class MonteCarlo(_Simulation):

    def __init__(self, limit_state_obj=None, distribution_obj=None, n_sim=None, n_tasks=1, random_state=None):
        self.distribution_obj = distribution_obj
        self.n_sim = n_sim
        self.n_tasks = n_tasks
        self.limit_state_obj = limit_state_obj
        self.random_state = random_state
        self.pf = None

        self.distribution_obj.random_state = random_state

        super().__init__(limit_state_obj=limit_state_obj,
                         distribution_obj=distribution_obj,
                         n_sim=n_sim, n_tasks=n_tasks, random_state=random_state)
        
    # Calculate the distance on the manifold
    def run(self):

        x = self.distribution_obj.rvs(n_sim=self.n_sim)
        self.limit_state_obj.run(X=x)
        g = self.limit_state_obj.g
        num_total = len(g)
        num_failure = sum(I < 0 for I in g)

        self.pf = num_failure/num_total

