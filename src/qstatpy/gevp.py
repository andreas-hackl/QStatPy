#!/usr/bin/env python3
import numpy as np
import scipy.linalg as LA

def normalize(W):
    for j in range(W.shape[1]):
        W[:,j] *= np.sign(W[0,j])
        W[:,j] /= np.linalg.norm(W[:,j])
    return W

def eig(Ct0, Ct):
    L = np.linalg.inv(Ct0) @ Ct
    if np.isfinite(L).all():
        V, W = LA.eig(Ct, Ct0)
        idx = V.argsort()[::-1]
        V = V[idx]
        W = W[:,idx]
        W = normalize(W)
    else:
        V = np.ones((L.shape[0],), L.dtype) * np.nan
        W = np.ones(L.shape, L.dtype) * np.nan

    return V, W


def normalize_i(W, tr):
    Wn = np.zeros_like(W)
    for i in range(W.shape[0]):
        Wn[:,i,:] = W[:,i,:]/np.linalg.norm(W[:,i,tr])
    return Wn

def normalize_j(W, tr):
    Wn = np.zeros_like(W)
    for i in range(W.shape[0]):
        Wn[i,:,:] = W[i,:,:]/np.linalg.norm(W[i,:,tr])
    return Wn

def measure(Wt, tr):
    r = 0
    for i in range(3):
        r += (np.linalg.norm(Wt[:,i,tr]) - 1)**2
        r += (np.linalg.norm(Wt[i,:,tr]) - 1)**2
    return r

def prepare(W, tr, tol=1e-12):
    Wt = W.copy()

    r = measure(Wt, tr)
    while r > tol:
        Wt = normalize_j(Wt, tr)
        Wt = normalize_i(Wt, tr)
        r = measure(Wt, tr)

    return Wt


class GEVP:

    def __init__(self, db, ctags, t0func, tag, osize=None):

        if len(ctags.shape) != 2 or len(set(ctags.shape)) != 1:
            raise ValueError(f"ctags has to have the shape (N,N) not {ctags.shape}")

        self.ctags = ctags
        self.nmodes = ctags.shape[0]
        self.gtag = tag
        self.t0func = t0func

        if osize is None:
            self.osize = self.nmodes
        else:
            self.osize = osize

        self.db = db
        self.create_data()

        # time mapping
        self.nt = db.mean(self.gtag, 'jk_mean').shape[2]

        tdata = np.arange(self.nt, dtype=int)
        self.t0map = t0func(tdata)

        tmask = self.t0map >= 0
        self.tdata = tdata[tmask]
        self.t0map = self.t0map[tmask]

        self.write_meta_data()

    def create_data(self):
        self.db.combine((self.gtag, 'jk_mean'), [(tag, 'jk_mean') for tag in self.ctags.flatten()], lambda v: np.array([[v[(tagij, 'jk_mean')] for tagij in tagi] for tagi in self.ctags]))

    def write_meta_data(self):
        self.db.add_data(self.nmodes, self.gtag, 'META', 'NMODES')
        self.db.add_data(self.tdata, self.gtag, 'META', 'TDATA')
        self.db.add_data(self.t0map, self.gtag, 'META', 'T0MAP')

    def __str__(self):
        s = f'{self.nmodes}x{self.nmodes} GEVP\n\n'
        s += f'nt = {self.nt}\n'

        if len(set(self.t0map)) == 1:
            s += f'with constant t0={self.t0map[0]}\n'
        if len(set(self.tdata - self.t0map)) == 1:
            s += f'with constant dt={self.tdata[0]-self.t0map[0]}\n'
    
        return s

    def emp_sample(self, y, cfg_tag):
        N = len(self.t0map)-1
        em = {f"emp_{i}": np.zeros((N,),dtype=np.double) for i in range(self.nmodes)}
        ev = {f"evec_{i}:{j}": np.zeros((N,), dtype=np.cdouble) for i in range(self.nmodes) for j in range(self.nmodes)}
        Wt = np.zeros((self.nmodes, self.nmodes, N), dtype=np.cdouble)

        for k, (t0, t) in enumerate(zip(self.t0map[:-1], self.tdata[:-1])):
            Ct0 = y[:,:,t0]
            Ct  = y[:,:,t]
            Cta = y[:,:,t+1]

            V0, W0 = eig(Ct0, Ct)
            V1, W1 = eig(Ct0, Cta)

            emp = np.log(V0/V1).real
            Wt[:,:,k] = W0

            for i in range(self.nmodes):
                em[f"emp_{i}"][k] = emp[i]
                for j in range(self.nmodes):
                    ev[f"evec_{i}:{j}"][k] = W0[j,i]

        self.db.add_data(Wt, self.gtag, "W", cfg_tag)

        for i in range(self.nmodes):
            self.db.add_data(em[f"emp_{i}"], self.gtag, f"emp_{i}", cfg_tag)
            for j in range(self.nmodes):
                self.db.add_data(ev[f"evec_{i}:{j}"], self.gtag, f"evec_{i}:{j}", cfg_tag)

    def create_emp(self):
        d = self.db.get_data(self.gtag, 'jk_mean')

        for cfg_tag, y in d.items():
            self.emp_sample(y, cfg_tag)


    def rebasing(self, idx, tag, n=None, t0func=None):
        if n is None:
            n = self.nmodes
        if t0func is None:
            t0func = self.t0func

        def trans_matrix(v):
            W = v[(self.gtag, 'W')][:,:,idx]
            return W

        self.db.combine((self.gtag, f'trans_mat_{idx}'), [(self.gtag, "W")], trans_matrix)

        def basis_transformation(v):
            W = v[(self.gtag, f'trans_mat_{idx}')]
            Winv = np.linalg.inv(W)
            C = v[(self.gtag, 'jk_mean')]
            Cn = np.zeros_like(C)
            for t in range(self.nt):
                Cn[:,:,t] = Winv @ C[:,:,t] @ W
            return Cn

        self.db.combine((self.gtag, f"jk_mean_diag_idx_{idx}"), [(self.gtag, 'jk_mean'), (self.gtag, f"trans_mat_{idx}")], basis_transformation)

        data = self.db.get_data(self.gtag, f"jk_mean_diag_idx_{idx}")
        for i in range(self.nmodes):
            for j in range(self.nmodes):
                self.db.add_data({cfg: v[i,j,:] for cfg, v in data.items()}, f"{tag}_{i}-{tag}_{j}", 'jk_mean')


        ctags = np.zeros((n,n), dtype=object)
        for i in range(n):
            for j in range(n):
                ctags[i,j] = f"{tag}_{i}-{tag}_{j}"

        rgevp = GEVP(self.db, ctags, t0func, tag, osize=self.osize)
        rgevp.create_emp()

        def prior_transformation(v):
            W = v[(self.gtag, f'trans_mat_{idx}')][:,:n]
            Wp = v[(self.gtag, 'prior_transf')]
            return Wp @ W

        if self.db.get_data(self.gtag, 'prior_transf') is not None:
            self.db.combine((tag, 'prior_transf'), [(self.gtag, f"trans_mat_{idx}"), (self.gtag, 'prior_transf')], prior_transformation)
        else:
            self.db.combine((tag, 'prior_transf'), [(self.gtag, f"trans_mat_{idx}")], lambda v: v[(self.gtag, f"trans_mat_{idx}")][:,:n])


        def mult(v):
            P = v[(tag, 'prior_transf')]
            W = v[(tag, 'W')]
            Wtrans = np.zeros((P.shape[0], W.shape[1], W.shape[2]), dtype=np.cdouble)
            for t in range(W.shape[2]):
                Wtrans[:,:,t] = normalize(P @ W[:,:,t])
            return Wtrans
            
        self.db.combine((tag, 'original_W'), [(tag, 'W'), (tag, 'prior_transf')], mult)

        data = self.db.get_data(tag, 'original_W')
        for i in range(n):
            for j in range(self.osize):
                
                self.db.add_data({cfg: v[j,i,:] for cfg, v in data.items()}, tag, f"oevec_{i}:{j}") 

        return rgevp
    
    def create_normalized_eigenstates(self, tr, tol=1e-12):
        ttags = [('gevp', f'evec_{i}:{j}') for i in range(self.nmodes) for j in range(self.nmodes)]
        def wrapper(v):
            M = np.array([[v[ttags[i*self.nmodes+j]] for j in range(self.nmodes)] for i in range(self.nmodes)])
            return prepare(M, tr, tol=tol)
        
        self.db.combine(('gevp', 'Wt'), ttags, wrapper)

        for i in range(self.nmodes):
            for j in range(self.nmodes):
                self.db.estimate(('gevp', 'Wt'), ("gevp", f"Wt_{i}:{j}"), lambda v: np.abs(v[i,j,:]))

    

