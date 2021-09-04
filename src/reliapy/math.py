import numpy as np
import scipy as sp


def spectral_decomposition(Cz):
    """
    Spectral decomposition of the correlation matrix.

    **Input:**
    * **Cz** (`ndarray`)
        Correlation matrix.

    **Output:**
    * **Jyz** (`ndarray`)
        Jacobian matrix for the transformation from Z to Y.

    * **Jzy** (`ndarray`)
        Jacobian matrix for the transformation from Y to Z.

    """

    L, A_ = sp.linalg.eig(Cz)
    Lsqrt = np.sqrt(np.real(L))
    A = A_ @ np.diag(np.reciprocal(Lsqrt))

    Jyz = A.T
    Jzy = np.linalg.inv(A.T)

    return Jyz, Jzy


def cholesky_decomposition(Cz):
    """
    Cholesky decomposition of the correlation matrix. This method is typically used for not full-matrices.

    **Input:**
    * **Cz** (`ndarray`)
        Correlation matrix.

    **Output:**
    * **Jyz** (`ndarray`)
        Jacobian matrix for the transformation from Z to Y.

    * **Jzy** (`ndarray`)
        Jacobian matrix for the transformation from Y to Z.

    """

    L = np.linalg.cholesky(Cz)

    Jyz = np.linalg.inv(L)
    Jzy = L

    return Jyz, Jzy
