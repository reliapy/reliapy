from reliapy._messages import *
from reliapy.math import phi_pdf, nataf, transform_xz
import numpy as np


class JointDistribution:
    """
    ``JointDistribution`` is a class implementing the elements of a joint distribution that are useful for reliability.

     **Input:**
    * **marginal** (`list`)
        A list of objects of marginal distribution.

    * **correlation** (`ndarray`)
        Correlation matrix.

    * **random_state** (`float`, `int`)
        Random seed.

    * **decomposition** (`str`)
        Decomposition of the correlation method: `spectral` or `cholesky`.

    **Attributes:**

    * **marginal** (`list`)
        A list of objects of marginal distribution.

    * **correlation** (`ndarray`)
        Correlation matrix.

    * **nrv** (`int`)
        Number of random variables.

    * **random_state** (`float`, `int`)
        Random seed.

    * **decomposition** (`str`)
        Decomposition of the correlation method: `spectral` or `cholesky`.

    * **mean** (`ndarray`)
        Array of means.

    * **std** (`ndarray`)
        Array of standard deviations.

    """

    def __init__(self, marginal=None, Cx=None, random_state=None, decomposition='spectral', correlation_z='approx'):

        if not isinstance(marginal, list):
            type_error('distributions', 'list')

        self.marginal = marginal
        self.Cx = Cx
        self.nrv = len(marginal)
        self.random_state = random_state
        self.decomposition = decomposition

        if correlation_z == 'approx':
            self.Cz = self.Cx
        elif correlation_z == 'nataf':
            self.Cz = nataf(self.Cx)
        else:
            not_implemented_error()

        mean = []
        std = []
        for i in range(self.nrv):
            m = marginal[i].stats[0]
            s = np.sqrt(marginal[i].stats[1])
            mean.append(m)
            std.append(s)

        self.mean = np.array(mean)
        self.std = np.array(std)

    def joint_pdf(self, X):
        """
        Joint PDF using the Nataf model.

        **Input:**
        * **X** (`float`)
            Argument.

        **Output**
        * **joint_pdf_val** (`float`)
            Joint PDF for `X`.
        """

        # Check is the input is consistent with the proabbility distribution.
        if len(X) != self.nrv:
            shape_error('X')

        # Get the Jacobians.
        Jxz, Jzx, M_eq, S_eq = transform_xz(X, distributions=self)

        # Transform the input from X to Z.
        Z = Jzx @ (X - M_eq)

        # Start the iterative process.
        f_prod = 1
        phi_prod = 1
        for i in range(self.nrv):
            f = self.marginal[i].pdf(X[i])
            f_prod = f_prod * f

            phi = phi_pdf(Z[i])
            phi_prod = phi_prod * phi

        phi_multi = phi_pdf(Z, corr=self.Cz)
        joint_pdf_val = phi_multi * (f_prod / phi_prod)

        return joint_pdf_val
