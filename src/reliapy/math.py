import numpy as np
import scipy as sp
from scipy.stats import norm
import copy


def phi_pdf(X):
    """
    Standard normal PDF.

    **Input:**
    * **X** (`float`)
        Argument.

    **Output**
        Standard normal PDF of X.
    """
    return norm.pdf(X, loc=0, scale=1)


def phi_cdf(X):
    """
    Standard normal CDF.

    **Input:**
    * **X** (`float`)
        Argument.

    **Output**
        Standard normal CDF of X.
    """
    return norm.cdf(X, loc=0, scale=1)


def phi_icdf(q):
    """
    Inverse of the standard normal CDF.

    **Input:**
    * **q** (`float`)
        Probability.

    **Output**
        Inverse value for the standard normal CDF.
    """
    return norm.ppf(q, loc=0, scale=1)


def pf2beta(pf):
    """
    Convert pf (probability of failure) into beta (reliability index).

    **Input:**
    * **pf** (`float`)
        Probability of failure.

    **Output**
    * **beta** (`float`)
        Reliability index.
    """
    beta = -norm.ppf(pf, loc=0, scale=1)
    return beta


def beta2pf(beta):
    """
    Convert beta (reliability index) into pf (probability of failure).

    **Input:**
    * **beta** (`float`)
        Reliability index.

    **Output**
    * **pf** (`float`)
        Probability of failure.

    """
    pf = norm.cdf(-beta, loc=0, scale=1)
    return pf


def normal_equivalent(X, distribution=None):
    """
    Get the mean and standard deviation for the equivalent normal distribution.

    **Input:**
    * **X** (`float`)
        Argument.

    * **distribution** (`object`)
        Marginal probability distribution.

    **Output**
    * **mu_eq** (`float`)
        Equivalent mean.

    * **std_eq** (`float`)
        Equivalent standard deviation.

    """
    q = distribution.cdf(X=X)
    z = phi_icdf(q)
    zf = phi_pdf(z)
    xf = distribution.pdf(X)

    std_eq = zf / xf
    mu_eq = X - z * std_eq

    return mu_eq, std_eq


def transform_xz(X, distributions=None):
    """
    Get the Jacobian from X to Z and vice-versa.

    **Input:**
    * **X** (`float`)
        Argument.

    * **distribution** (`object`)
        Marginal probability distribution.

    **Output**
    * **Jxz** (`ndarray`)
        Jacobian from Z to X.

    * **Jzx** (`ndarray`)
        Jacobian from X to Z.

    * **M_eq** (`float`)
        Array of equivalent means.

    * **D_eq** (`float`)
        Array of equivalent standard deviations.

    """
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
    """
    Numerical gradient of a given function `fun` using finite differences.

    **Input:**
    * **X** (`float`)
        Argument.

    * **fun** (`callable`)
        Function.

    **Output**
    * **gradient** (`ndarray`)
        Gradient of `fun`.

    """
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
