from reliapy.math import *
from reliapy.monte_carlo._simulation import _Simulation


class MonteCarlo(_Simulation):
    """
    ``MonteCarlo`` is a class implementing the crude Monte Carlo simulation, and it is a child class of ``_Simulation``.

    **Input:**
    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    **Attributes:**

    * **limit_state_obj** (`object`)
        Object of ``LimitState``.

    * **distribution_obj** (`object`)
        Object of ``JointDistribution``.

    * **n_sim** (`int`)
        Number of simulations.

    * **n_tasks** (`int`)
        Number of threads in parallel computing.

    * **random_state** (`int`)
        Random seed for the random number generator.

    * **pf** (`float`)
        Probability of failure.

    * **beta** (`float`)
        Reliability index.

    """

    def __init__(self, limit_state_obj=None, distribution_obj=None, n_sim=None, n_tasks=1, random_state=None):
        self.distribution_obj = distribution_obj
        self.n_sim = n_sim
        self.n_tasks = n_tasks
        self.limit_state_obj = limit_state_obj
        self.random_state = random_state
        self.pf = None
        self.beta = None

        # set the random state of the ``JointDistribution`` object.
        self.distribution_obj.random_state = random_state

        super().__init__(limit_state_obj=limit_state_obj,
                         distribution_obj=distribution_obj,
                         n_sim=n_sim, n_tasks=n_tasks, random_state=random_state)

    def run(self):
        """
        Run Monte Carlo simulation.
        """

        # Get `n_sim` random samples.
        x = self.distribution_obj.rvs(n_sim=self.n_sim)

        # Evaluate the limit state functions for the random samples.
        self.limit_state_obj.run(X=x)
        g = self.limit_state_obj.g

        # Get the number of samples in the failure domain.
        num_failure = sum(I < 0 for I in g)

        # Compute the probability of failure.
        self.pf = num_failure/self.n_sim
        self.beta = pf2beta(self.pf)

