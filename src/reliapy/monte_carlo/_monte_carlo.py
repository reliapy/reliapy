from reliapy.math import *
from reliapy.sampling import LHS, Random


class MonteCarlo:
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

    def __init__(self, limit_state_obj=None, sampling_obj=None, n_sim=None, n_tasks=1, random_state=None):
        self.sampling_obj = sampling_obj
        self.n_sim = n_sim
        self.n_tasks = n_tasks
        self.limit_state_obj = limit_state_obj
        self.random_state = random_state
        self.pf = None
        self.beta = None

        # set the random state of the ``JointDistribution`` object.
        self.sampling_obj.random_state = random_state

    def run(self):
        """
        Run Monte Carlo simulation.
        """

        # Get `n_sim` random samples.
        if isinstance(self.sampling_obj, LHS) or isinstance(self.sampling_obj, Random):
            x = self.sampling_obj.rvs(n_sim=self.n_sim)

            # Evaluate the limit state functions for the random samples.
            self.limit_state_obj.run(X=x)
            g = self.limit_state_obj.g

            # Get the number of samples in the failure domain.
            num_failure = sum(I < 0 for I in g)

            # Compute the probability of failure.
            self.pf = num_failure / self.n_sim

        else:
            x, x_ = self.sampling_obj.rvs(n_sim=self.n_sim)

            # Evaluate the limit state functions for the random samples.
            self.limit_state_obj.run(X=x)
            g = self.limit_state_obj.g

            self.limit_state_obj.run(X=x_)
            g_ = self.limit_state_obj.g

            # Get the number of samples in the failure domain.
            num_failure = sum(I < 0 for I in g)
            num_failure_ = sum(I < 0 for I in g_)

            pf = num_failure / self.n_sim
            pf_ = num_failure_ / self.n_sim

            # Compute the probability of failure.
            self.pf = (pf + pf_) / 2

        self.beta = pf2beta(self.pf)

