import scipy as sp
import numpy as np
from reliapy._messages import *
from reliapy.monte_carlo import _Simulation


class MonteCarlo(_Simulation):

    def __init__(self, limit_state_obj=None, distributions=None, n_sim=None, n_tasks=1, random_state=None):
        self.distributions = distributions
        self.n_sim = n_sim
        self.n_tasks = n_tasks
        self.limit_state_obj = limit_state_obj
        self.random_state = random_state

        super().__init__(limit_state_object=limit_state_obj,
                         distributions=distributions,
                         n_sim=n_sim, n_tasks=n_tasks, random_state=random_state)
        
    # Calculate the distance on the manifold
    def run(self):
        template_error()

