#!/usr/bin/env python3

import numpy as np
from scipy.stats import chi2
import time
from scipy.optimize import OptimizeResult
from qstatpy.fit.general import jacobian

# abbreviations
T = np.transpose
Dt = np.dot 


class LevenbergMarquardt:

    def __init__(self, beta0, model, ydata, W, maxiter=1000, eps1=1e-7, eps2=1e-7,
                 eps3=1e-5, eps4=1e-5, alpha0=0.01, eps_grad=1e-12, Aup=11, Adown=9, 
                 nu=2, fjac=None):

        self.beta0 = np.array(beta0, dtype=np.double)
        self.beta = np.array(beta0, dtype=np.double)

        self.model = model
        self.ydata = np.array(ydata, dtype=np.double)
        self.W = np.array(W, dtype=np.double)

        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.eps4 = eps4

        self.eps_grad = eps_grad
        
        if fjac is None:
            self.fjac = lambda x: jacobian(self.model, x, eps=self.eps_grad)
        else:
            self.fjac = fjac

        # Calculate the first jacobian
        self.jac = self.fjac(self.beta0)

        self.alpha0 = alpha0
        self.alpha = alpha0
        self.maxiter = maxiter
        self.Aup = Aup
        self.Adown = Adown

        self.nu = nu
        self.conv = False
        self.nit = 0

    #######
    ## Function determining the convergence of the algorithm
    #######

    def conv_grad(self):
        dy = self.ydata - self.model(self.beta)
        val = max(abs(Dt(T(self.jac), Dt(self.W, dy))))
        return val < self.eps1

    def conv_param(self, step):
        val = max(abs(step/self.beta))
        return val < self.eps2

    def conv_chi2(self):
        return self.chi2()/(self.ydata.shape[0] - self.beta0.shape[0]+1) < self.eps3

    def isconv(self, step):
        return self.conv_grad() or self.conv_chi2() or self.conv_param(step)

    #######
    ## Useful functions in the optimization steps
    #######

    def update_jac(self):
        self.jac = self.fjac(self.beta)

    def chi2(self, step=0):
        dy = self.ydata - self.model(self.beta + step)
        return Dt(T(dy), Dt(self.W, dy))

    def dchi2(self, step):
        return self.chi2() - self.chi2(step)

    def jwj(self):
        # J^T W J is a frequently occuring term
        return Dt(T(self.jac), Dt(self.W, self.jac))

    def set_timer(self):
        self.TSTART = time.time()

    def message(self):
        print(f"step {self.nit:5d}\tred_chi2 = {self.redchi2():.5e}\t{time.time() - self.TSTART:.5f} [sec]")

    #######
    ## LEVENBERG-MARQUARDT UPDATE
    #######

    def _lm_rho(self, step, JWJ, rhs):
        dchi_ = self.dchi2(step)
        den = Dt(T(step), self.alpha*Dt(np.diag(JWJ), step) + rhs)
        return dchi_/den

    def _lm_update(self):
        JWJ = self.jwj()

        # solve A x = v for x
        A = JWJ + self.alpha * np.diag(np.diag(JWJ))
        dy = self.ydata - self.model(self.beta)
        v = Dt(T(self.jac), Dt(self.W, dy))
        step = np.linalg.solve(A, v)

        # compute rhoi
        rhoi = self._lm_rho(step, JWJ, v)

        if self.isconv(step):
            self.conv = True

        if rhoi > self.eps4:
            self.beta += step
            self.update_jac()
            self.alpha = max([self.alpha/self.Adown, 1e-7])
        else:
            self.alpha = min([self.alpha*self.Aup, 1e7])

        self.nit += 1

    def _lm(self, verbose=False):
        self.set_timer()
        while self.nit < self.maxiter and not self.conv:
            self._lm_update()
            if verbose and self.nit % 10 == 0: self.message()

    ########
    ## NIELSON UPDATE
    ########

    def _nielson_rho(self, step, rhs):
        dchi_ = self.dchi2(step)
        den = Dt(T(step), self.alpha * step + rhs)
        return dchi_/den

    def _nielson_update(self):
        JWJ = self.jwj()

        # Solve A x = v for x 
        A = JWJ + self.alpha * np.eye(JWJ.shape[0])
        dy = self.ydata - self.model(self.beta)
        v = Dt(T(self.jac), Dt(self.W, dy))
        step = np.linalg.solve(A, v)

        rhoi = self._nielson_rho(step, v)

        if self.isconv(step):
            self.conv = True

        if rhoi > self.eps4:
            self.beta += step 
            self.update_jac()
            self.alpha *= max([1/3, 1 - (2*rhoi - 1)**3])
            self.nu = 2
        else:
            self.alpha *= self.nu
            self.nu *= 2

        self.nit += 1

    def _nielson(self, verbose=False):
        self.set_timer()

        # Startup
        JWJ = self.jwj()
        self.alpha = self.alpha0 * max(np.diag(JWJ))

        while self.nit < self.maxiter and not self.conv:
            self._nielson_update()
            if verbose and self.nit % 10 == 0: self.message()

    #######
    ## OUTPUT FUNCTIONS
    #######

    def solve(self, mode='nielson', verbose=False):
        if mode == 'lm':
            self._lm(verbose=verbose)
        elif mode == 'nielson':
            self._nielson(verbose=verbose)
        else:
            raise ValueError(f'invalid mode: {mode}')

        sol = OptimizeResult(x=self.beta, success=self.conv, status=0, fun=self.chi2())

        return sol


def levenberg_marquardt(model, p0, ydata, W, **kwargs):
    LMobj = LevenbergMarquardt(p0, model, ydata, W, **kwargs)
    return LMobj.solve()
        

        

