from reliapy._messages import *


class _Simulation:
    """
    ``_Simulation`` is a parent class (template) for the simulation methods.

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
        self.limit_state_function = limit_state_obj
        self.random_state = random_state
        self.pf = None
        self.beta = None

    def run(self):
        """
        Run Simulation (template).
        """
        template_error()

