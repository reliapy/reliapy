from reliapy._messages import *


class Distribution:

    def __init__(self, distributions=None, correlation=None, n_sim=None, n_tasks=1, random_state=None):

        if not isinstance(distributions, list):
            type_error('distributions', 'list')

        self.distributions = distributions
        self.n_sim = n_sim
        self.n_tasks = n_tasks
        self.random_state = random_state
        self.correlation = correlation
        self.nrv = len(distributions)

    # Calculate the distance on the manifold
    def run(self):
        template_error()
