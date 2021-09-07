from reliapy._messages import *


class _Continuous:
    """
    ``_Continuous`` is a parent class (template) for the classes implementing the 1D constinuous distributions.

    **Input:**
    * **loc** (`float`)
        Location of the random variable (as in scipy.stats).

    * **scale** (`float`)
        Scale of the random variable (as in scipy.stats).

     * **random_state** (`float`, `int`)
        Random seed for the random number generator.

    **Attributes:**

    * **loc** (`float`)
        Location of the random variable (as in scipy.stats).

    * **scale** (`float`)
        Scale of the random variable (as in scipy.stats).

    * **random_state** (`float`, `int`)
        Random seed for the random number generator.

    * **central_moments** (`ndarray`)
        Array with central moments.

    """

    def __init__(self):
        pass

    def pdf(self, X=None):
        template_error()

    def cdf(self, X=None):
        template_error()

    def icdf(self, X=None):
        template_error()

    def moment(self, n=1):
        template_error()

    def rvs(self, n_sim=1):
        template_error()
