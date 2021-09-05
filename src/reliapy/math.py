import numpy as np
import scipy as sp
from scipy.stats import norm
import copy


def phi_pdf(X):
    return norm.pdf(X, loc=0, scale=1)


def phi_cdf(X):
    return norm.cdf(X, loc=0, scale=1)


def phi_icdf(q):
    return norm.ppf(q, loc=0, scale=1)


def normal_equivalent(X, distribution=None):
    q = distribution.cdf(X=X)
    z = phi_icdf(q)
    zf = phi_pdf(z)
    xf = distribution.pdf(X)

    std_eq = zf / xf
    mu_eq = X - z * std_eq

    return mu_eq, std_eq


def transform_xz(X, distributions=None):
    nrv = len(X)

    M_eq = []
    D_eq = []
    for i in range(nrv):
        mu_eq, std_eq = normal_equivalent(X[i], distribution=distributions.marginal[i])
        M_eq.append(mu_eq)
        D_eq.append(std_eq)

    M_eq = np.array(M_eq)
    D_eq = np.array(D_eq)

    Jzx = np.diag(D_eq)
    Jxz = np.diag(np.reciprocal(D_eq))

    return Jxz, Jzx, M_eq, D_eq


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


def numerical_gradient(X, fun):
    nrv = len(X)
    h = 1e-6
    gradient = []
    for i in range(nrv):
        x0 = copy.copy(X)
        x0[i] = x0[i] + h
        f0 = fun(x0)

        x1 = copy.copy(X)
        x1[i] = x1[i] - h
        f1 = fun(x1)

        df = (f0 - f1) / (2 * h)
        gradient.append(df)

    return gradient
