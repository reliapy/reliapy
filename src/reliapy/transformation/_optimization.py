import scipy as sp
import numpy as np
import copy
from reliapy._messages import *


class Optimization:

    def __init__(self, limit_state_obj=None, distribution_obj=None):
        self.distribution_obj = distribution_obj
        self.limit_state_obj = limit_state_obj

    def HLRF(self, tol=1e-3, max_iter=20):

        if tol < 0:
            value_error('tol')

        if max_iter < 1:
            value_error('max_iter')

        mean = self.distribution_obj.mean
        std = self.distribution_obj.std

        y = np.zeros(len(mean))
        error = 10000
        itera = 0
        while error > tol and itera < max_iter:
            y_before = copy.copy(y)
            x = mean + y * std
            dxdy = std

            g = self.limit_state_obj.function(x)
            dgdx = self.limit_state_obj.gradient(x)
            dgdy = dgdx * dxdy

            c = (np.dot(dgdy, y) - g) / (np.linalg.norm(dgdy) ** 2)
            y = c * dgdy
            error = np.linalg.norm(y - y_before)
            itera = itera + 1

        return y

    def iHLRF(self, a=0.1, b=0.5, gamma=2, tol=1e-3, tol_1=1e-3, tol_2=1e-3, max_iter=20):

        if gamma < 1:
            value_error('gamma')

        if tol_1 < 0:
            value_error('tol_1')

        if tol_2 < 0:
            value_error('tol_2')

        if max_iter < 1:
            value_error('max_iter')

        if a < 0 or a > 1:
            value_error('a')

        if b < 0 or b > 1:
            value_error('b')

        mean = self.distribution_obj.mean
        std = self.distribution_obj.std

        y = np.ones(len(mean))

        x0 = mean + y * std
        g0 = self.limit_state_obj.function(x0)
        tol_2 = tol_2 * abs(g0)

        itera = 0
        while itera < max_iter:
            y_before = copy.copy(y)
            x = mean + y * std
            dxdy = std

            g = self.limit_state_obj.function(x)
            dgdx = self.limit_state_obj.gradient(x)
            dgdy = dgdx * dxdy

            error_1 = 1 - abs(np.dot(dgdy, y)) / (np.linalg.norm(dgdy) * np.linalg.norm(y))
            error_2 = np.linalg.norm(g)

            if error_1 < tol_1 and error_2 < tol_2:
                break

            c = (np.dot(dgdy, y) - g) / (np.linalg.norm(dgdy) ** 2)
            dk = c * dgdy - y

            if error_2 >= tol_2:
                v0 = np.linalg.norm(y) / np.linalg.norm(dgdy)
                v1 = 0.5 * (np.linalg.norm(y + dk) ** 2) / abs(g)
                ck = gamma * max(v0, v1)
            else:
                v0 = np.linalg.norm(y) / np.linalg.norm(dgdy)
                ck = gamma * v0

            # linear search.

            n = 0
            while True:
                y0 = y + (b ** n) * dk
                m0 = self._merit(y0, ck, mean, std)
                m1 = self._merit(y, ck, mean, std)
                dm = m0 - m1

                gm = self._grad_merit(y, ck, mean, std)
                dgm = -a * (b ** n) * np.linalg.norm(gm)

                if dm <= dgm:
                    break

                n = n + 1

            y = y + (b ** n) * dk
            error = np.linalg.norm(y - y_before)
            if error < tol:
                break
            #print(y)
            #print(error_1, error_2, itera)
            itera = itera + 1

        return y

    def _merit(self, y, c, mean, std):

        x = mean + y * std
        g = self.limit_state_obj.function(x)
        m = 0.5 * np.linalg.norm(y) ** 2  + c * abs(g)

        return m

    def _grad_merit(self, y, c, mean, std):

        nrv = len(y)
        h = 1e-6
        gradient = []
        for i in range(nrv):
            y0 = copy.copy(y)
            y0[i] = y0[i] + h
            f0 = self._merit(y0, c, mean, std)

            y1 = copy.copy(y)
            y1[i] = y1[i] - h
            f1 = self._merit(y1, c, mean, std)

            df = (f0 - f1) / (2 * h)
            gradient.append(df)

        return gradient

