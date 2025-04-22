#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

import qstatpy
import warnings


class Fitter:

    def __init__(self, db, in_tag, out_tag, func, p0, frange, x=None, correlated_fit=False, eps=1.0):
        self.func = func
        self.p0 = p0
        self.db = db

        self.eps = eps
        self.frange = frange

        self.in_tag = in_tag
        self.out_tag = out_tag

        data = db.get_data(*in_tag)
        
        self.nt = np.array(list(data.values())[0]).shape[0]
        if x is None:
            self.x_ = np.arange(frange[0], frange[1]+1)
            self.x = np.arange(self.nt)
        else:
            self.x_ = x[frange[0] : frange[1]+1]
            self.x = x
        

        mask = np.zeros((self.nt, ), dtype=bool)
        mask[frange[0] : frange[1]+1] = True

        self.tmp_tag = (in_tag[0], 'tmp_masked')

        db.operation(in_tag, lambda c: c[mask], out_tag=self.tmp_tag)

        self.mean = db.mean(*self.tmp_tag)
        cov = db.cov(*self.tmp_tag)
        std = np.sqrt(np.diag(cov))
        if correlated_fit:
            self.W = np.linalg.inv(cov)
        else:
            self.W = np.diag(1/(std**2))

    def chi2_cost(self, p, y):
        dy = self.func(self.x_, p) - y
        return np.dot(np.conjugate(dy), np.dot(self.W, dy)).real
    

    def model(self, p):
        return self.func(self.x_, p)
    
    def fitting_lm(self, y, maxiter=1000, fjac=None):
        sol = qstatpy.fit.optimizer.levenberg_marquardt(self.model, self.p0, y, self.W, maxiter=maxiter, fjac=fjac)
        if not sol.success:
            print(sol)
            raise RuntimeError()
        return sol.x

    def fitting_scipy(self, y, method='Nelder-Mead', maxiter=1000):
        sol = opt.minimize(self.chi2_cost, self.p0, args=(y,), method=method, options={'maxiter': maxiter})
        if not sol.success:
            print(sol)
            raise RuntimeError()
        return sol.x
    
    def extrapolation_check(self):
        mean, std = qstatpy.fit.extrapolation_check(self.db, self.func, self.in_tag, self.out_tag, 
                                                   (self.out_tag[0], self.out_tag[1]+'_extrapolation'), self.x[self.frange[0]-1], self.frange[0]-1,
                                                   np.ones_like(self.p0, dtype=bool), eps=self.eps)
        return mean/std
    

    def solve(self, mode='lm', method='Nelder-Mead', maxiter=1000, fjac=None):
        if mode == 'lm':
            fitfunc = lambda y: self.fitting_lm(y, maxiter=maxiter, fjac=None)
        elif mode == 'scipy':
            fitfunc = lambda y: self.fitting_scipy(y, method=method, maxiter=maxiter)
        else:
            raise ValueError()
        

        fmean = fitfunc(self.mean)
        self.p0 = fmean

        fcov = self.db.estimate(self.tmp_tag, self.out_tag, f=fitfunc, eps=self.eps, fmean=fmean)

        dof = self.mean.shape[0] - self.p0.shape[0] + 1
        chi2 = self.chi2_cost(fmean, self.mean)
    
        extrapol = self.extrapolation_check()

        sol = {
            'beta': fmean,
            'cov': fcov,
            'dof': dof,
            'chi2': chi2,
            'redchi2': chi2/dof,
            'p': 1 - stats.chi2.cdf(chi2, dof),
            'extrapolation_check': extrapol,
        }

        self.db.add_data(sol, self.out_tag[0], self.out_tag[1]+'_sol')
        self.db.remove(*self.tmp_tag)

        return sol


    



        
