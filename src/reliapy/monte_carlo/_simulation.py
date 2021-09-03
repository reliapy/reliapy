import scipy as sp
import numpy as np
from reliapy._messages import *


class _Simulation:

    def __init__(self, limit_state_obj=None, distributions=None, n_sim=None, n_tasks=1, random_state=None):
        self.distributions = distributions
        self.n_sim = n_sim
        self.n_tasks = n_tasks
        self.limit_state_function = limit_state_obj
        self.random_state = random_state

    # Calculate the distance on the manifold
    def run(self):
        template_error()

