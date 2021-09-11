import numpy as np
import scipy as sp
from scipy.stats import norm, multivariate_normal
from scipy.stats import multivariate_normal as multi_norm
import scipy.integrate as si
import copy
from reliapy._messages import *


def nataf(Cx, max_iter=5, tol=1e-10):
    """
    Nataf model for the transformation of the correlation matrix from the domain X to Z.

    **Input:**
    * **Cx** (`ndarray`)
        Correlation matrix in X.

    * **max_iter** (`int`)
        Maximum number of iterations.

    * **max_iter** (`int`)
        Maximum number of iterations.

    * **tol** (`float`)
        Error tolerance.

    **Output**
    * **Cz** (`ndarray`)
        Correlation matrix in Z.
    """
    h = 0.000000001
    rows, cols = np.shape(Cx)
    Cz = np.eye(rows)
    for i in np.arange(0, rows):
        for j in np.arange(i + 1, cols):
            # initial guess.
            rho_z = Cx[i, j]

            itera = 0
            while itera < max_iter:
                rho_h = rho_z
                err0 = _func_nataf(rho_h, Cx[i, j])

                if err0 < tol:
                    break

                rho_h = rho_z + h
                err1 = _func_nataf(rho_h, Cx[i, j])

                d_rho = (err1 - err0) / h
                rho_z = rho_z - d_rho

                if rho_z > 1:
                    rho_z = 1
                elif rho_z < 0:
                    rho_z = 0

                itera = itera + 1

            Cz[i, j] = rho_z
            Cz[j, i] = rho_z

    return Cz


def _func_nataf(rho_z, rho_x):
    """
    Private method used to find the correlation coefficients in Z.

    **Input:**
    * **rho_z** (`float`)
        Correlation coefficent in Z.

    * **rho_x** (`float`)
        Correlation coefficient in X.

    **Output**
    * **z** (`float`)
        Error between the estimated `rho_x` for a given `rho_z` and real value of `rho_x`.
    """
    rho_x_, _ = si.dblquad(lambda x, y: x * y * multi_norm.pdf((x, y), cov=[[1, rho_z], [rho_z, 1]]),
                           -np.inf, np.inf, -np.inf, np.inf)

    z = np.abs(rho_x_ - rho_x) ** 2

    return z


def phi_pdf(X, corr=None):
    """
    Standard normal PDF/Multivariate pdf.

    **Input:**
    * **X** (`float`)
        Argument.

    * **corr** (`ndarray`)
        Correlation matrix.

    **Output**
        Standard normal PDF of X.
    """
    norm_pdf = None
    if isinstance(X, int) or isinstance(X, float):
        norm_pdf = norm.pdf(X, loc=0, scale=1)
    else:
        if np.trace(corr) != len(X):
            shape_error(' X or corr ')
        else:
            norm_pdf = multivariate_normal.pdf(X, cov=corr)

    return norm_pdf


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

    Jxz = np.diag(D_eq)
    Jzx = np.diag(np.reciprocal(D_eq))

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
