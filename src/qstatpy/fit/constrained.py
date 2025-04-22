import qstatpy
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import warnings


class ConstrainedFitter:

    def __init__(self, db, in_tags, out_tag, funcs, p0, franges, nshared=0, x=None, correlated_fit=False, eps=1.0):
        self.db = db
        self.p0 = p0
        self.nshared = nshared
        self.N = len(in_tags)
        self.nfit = (p0.shape[0] - nshared)//self.N

        self.in_tags = in_tags
        self.out_tag = out_tag
        self.eps = eps

        if not isinstance(funcs, list):
            self.funcs = [funcs for i in range(self.N)]
        else:
            self.funcs = funcs

        data = [db.get_data(*in_tag) for in_tag in in_tags]
        nts = [np.array(list(d.values())[0]).shape[0] for d in data]
        masks = [np.zeros((nt, ), dtype=bool) for nt in nts]

        for i, frange in enumerate(franges):
            masks[i][frange[0]:frange[1]+1] = True

        self.tmp_tags = []
        self.mean = np.array([])
        self.Ws = []
        for i, in_tag in enumerate(in_tags):
            tmp_tag = (in_tag[0], f'tmp_masked_{i}')
            self.tmp_tags.append(tmp_tag)

            db.operation(in_tag, lambda c: c[masks[i]], out_tag=tmp_tag)

            mean = db.mean(*tmp_tag)
            std = db.std(*tmp_tag, eps=eps).real
            cov = db.cov(*tmp_tag, eps=eps).real

            if correlated_fit:
                W = np.linalg.inv(cov)
            else:
                W = np.diag(1/(std**2))

            self.mean = np.concatenate((self.mean, mean))
            self.Ws.append(W)

        self.tmp_tag_full = (out_tag[0], 'tmp_masked')
        db.combine(self.tmp_tag_full, self.tmp_tags, lambda v: np.concatenate(tuple([v[in_tag] for in_tag in self.tmp_tags])))

        self.masks = masks
        self.franges = franges
        self.xs = [np.arange(r[0], r[1]+1) for r in franges]
        self.xs_ = [np.arange(nt) for nt in nts]

        self.W = self.total_W()

    def total_W(self):
        sizes = [Wi.shape[0] for Wi in self.Ws]
        ns = [sum(sizes[:i]) for i in range(len(sizes))] + [sum(sizes)]

        W = np.zeros((ns[-1], ns[-1]), dtype=np.double)
        for j, Wj in enumerate(self.Ws):
            W[ns[j]:ns[j+1], ns[j]:ns[j+1]] = Wj
        return W

    def chi2_cost(self, p, y):
        pshared = p[:self.nshared]
        res = 0

        sizes = [len(x) for x in self.xs]
        commulative = [sum(sizes[:i]) for i in range(len(sizes))] + [sum(sizes)]
        for i, x in enumerate(self.xs):
            pi = np.concatenate((pshared, p[self.nshared+i*self.nfit:self.nshared+(i+1)*self.nfit]))
            dyi = self.funcs[i](x, pi) - y[commulative[i]:commulative[i+1]]
            res += np.dot(dyi, np.dot(self.Ws[i], dyi))

        return res

    def model(self, p):
        pshared = p[:self.nshared]
        v = np.array([])
        for i, x in enumerate(self.xs):
            pi = np.concatenate((pshared, p[self.nshared+i*self.nfit:self.nshared+(i+1)*self.nfit]))
            vi = self.funcs[i](x, pi)
            v = np.concatenate((v, vi))
        return v

    def y_full(self, y):
        yr = np.array([])
        for yi in y:
            yr = np.concatenate((yr, yi))
        return yr

    def fitting_lm(self, y, maxiter=1000, fjac=None):
        sol = qstatpy.fit.optimizer.levenberg_marquardt(self.model, self.p0, y, self.W, maxiter=maxiter, fjac=fjac)
        if not sol.success:
            print(sol)
            raise RuntimeError()
        return sol.x

    def fitting_scipy(self, y, method='Nelder-Mead', maxiter=1000):
        sol = opt.minimize(self.chi2_cost, self.p0, args=(y,), method=method, options={'maxiter':maxiter})
        if not sol.success:
            print(sol)
            raise RuntimeError()

    def extrapolation_check(self):
        ext = []
        for i, in_tag in enumerate(self.in_tags):
            pmask = np.zeros_like(self.p0, dtype=bool)
            pmask[:self.nshared] = True
            pmask[self.nshared+i*self.nfit:self.nshared+(i+1)*self.nfit] = True
            mean, std = qstatpy.fit.extrapolation_check(self.db, self.funcs[i], in_tag, self.out_tag, 
                                                       (self.out_tag[0], self.out_tag[1]+f'_extrapolation_{i}'),
                                                       self.xs_[i][self.franges[i][0]-1], self.franges[i][0]-1,
                                                       pmask, eps=self.eps)
            ext.append(mean/std)
        return ext
        

    def solve(self, mode='lm', method='Nelder-Mead', maxiter=1000, fjac=None):
        if mode == 'lm':
            fitfunc = lambda y: self.fitting_lm(y, maxiter=maxiter, fjac=fjac)
        elif mode == 'scipy':
            fitfunc = lambda y: self.fitting_scipy(y, method=method, maxiter=maxiter)

        fmean = fitfunc(self.mean)
        self.p0 = fmean
        
        fcov = self.db.estimate(self.tmp_tag_full, self.out_tag, f=fitfunc, eps=self.eps, fmean=fmean)

        dof = self.mean.shape[0] - self.p0.shape[0] +1
        chi2 = self.chi2_cost(fmean, self.mean)
        ext = self.extrapolation_check()

        sol = {
            'beta': fmean,
            'cov': fcov,
            'dof': dof,
            'chi2': chi2,
            'redchi2': chi2/dof,
            'p': 1 - stats.chi2.cdf(chi2, dof),
        }

        for i, in_tag in enumerate(self.in_tags):
            sol[f'extrapolation_check_{i}'] = ext[i] 

        self.db.add_data(sol, self.out_tag[0], self.out_tag[1]+'_sol')

        for tmp_tag in self.tmp_tags:
            self.db.remove(*tmp_tag)
        self.db.remove(*self.tmp_tag_full)

        return sol


            
            
