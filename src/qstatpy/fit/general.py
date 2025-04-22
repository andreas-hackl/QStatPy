#!/usr/bin/env python3

import numpy as np
import qstatpy



def approx_fprime(f, x0, eps=1e-12, args=()):
    return (f(x0 + eps, *args) - f(x0 - eps, *args))/(2*eps)

def approx_grad(f, x0, eps=1e-12, args=()):
    if isinstance(x0, (float, int)):
        return approx_fprime(f, x0, eps=eps, args=args)
    
    gradf = np.zeros(x0.shape, dtype=x0.dtype)

    for i, x in enumerate(x0):
        ei = np.zeros((x0.shape[0],), dtype=x0.dtype)
        ei[i] = eps
        gradf[i] = (f(x0 + ei, *args) - f(x0 - ei, *args))/(2*eps)
    return gradf

def jacobian(f, x0, eps=1e-12, args=()):
    f0 = f(x0, *args)
    ndim = x0.shape[0]
    mdim = f0.shape[0]

    jac = np.zeros((mdim, ndim), dtype=f0.dtype)

    for i in range(ndim):
        epsi = np.zeros_like(x0)
        epsi[i] = eps
        dfdpi = (f(x0 + epsi, *args) - f(x0 - epsi, *args))/(2*eps)
        jac[:,i] = dfdpi
    return jac


def errorband_jacobian(func, xvals, beta, cov, eps=1e-10):
    J = jacobian(lambda p: func(xvals, p), beta, eps=eps)
    return np.sqrt(np.diag(J @ cov @ np.transpose(J)))

# write new function
def errorband_jackknife(db, func, xvals, param_tag):    
    return np.sqrt(np.diag(db.estimate(param_tag, f=lambda p: func(xvals, p))))

def errorband(param0, *params):
    if type(param0) == qstatpy.Database:
        return errorband_jackknife(param0, *params)
    return errorband_jacobian(param0, *params)


def extrapolation_check(db, func, jk_tag, fit_tag, extra_tag, x0, idx0, param_mask, eps=1.0):
    def extrapol(v):
        beta = v[fit_tag]
        y = v[jk_tag]
        y_ex = func(x0, beta[param_mask])
        return np.array([y_ex - y[idx0]])
    db.combine(extra_tag, [jk_tag, fit_tag], f=extrapol)
    return db.mean(*extra_tag), db.std(*extra_tag)
