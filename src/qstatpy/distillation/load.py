import qstatpy
import numpy as np
import itertools as it
import multiprocessing

N_cpu = multiprocessing.cpu_count()


def load_single_eps(fn, N_dist=None):
    d_eps = qstatpy.gpt_io.get_data(fn+'/head.dat')
    tag_struct = next(iter(d_eps))
    nt = d_eps[tag_struct].shape[0]
    if N_dist is None:
        N_dist = int(np.round(len(d_eps)**(1/3),4))

    a_eps = np.zeros((nt, N_dist, N_dist, N_dist), dtype=np.cdouble)
    
    for l, n, m in it.product(range(N_dist), repeat=3):
        a_eps[:,l,n,m] = d_eps[f"eps_{l}_{n}_{m}"]

    return a_eps


def load_eps(src, momenta, N_dist=None):
    print(f"{qstatpy.timer.__str__()}Loading eps tensors", flush=True)

    eps_dict = {tuple(p): load_single_eps(f"{src}/pm_eps_{p[0]}.{p[1]}.{p[2]}", N_dist) for p in momenta}
    return eps_dict


def load_single_peramb(fn, N_dist=None):
    d_peramb = qstatpy.gpt_io.get_data(fn+'/head.dat')

    tag_struct = next(iter(d_peramb))
    prec = tag_struct.split('/')[1]
    t0 = int(tag_struct.split('/')[-1].split('_')[-1])
    nt = d_peramb[tag_struct].shape[0]
    if N_dist is None:
        N_dist = int(np.round(np.sqrt(len(d_peramb)/16), 4))

    a_peramb = np.zeros((nt, N_dist, N_dist, 4, 4), dtype=np.cdouble)
    for n, m in it.product(range(N_dist), repeat=2):
        for s0, s1 in it.product(range(4), repeat=2):
            a_peramb[:,n,m,s0,s1] = d_peramb[f"output/{prec}/n_{n}_{m}_s_{s0}_{s1}_t_{t0}"]

    return a_peramb

def load_single_seq(fn, N_dist=None):
    d_peramb = qstatpy.gpt_io.get_data(fn+'/head.dat')

    tag_struct = next(iter(d_peramb))
    prec = tag_struct.split('/')[1]
    t0 = int(tag_struct.split('/')[-1].split('_')[-2])
    ts = int(tag_struct.split('/')[-1].split('_')[-1])
    nt = d_peramb[tag_struct].shape[0]
    if N_dist is None:
        N_dist = int(np.round(np.sqrt(len(d_peramb)/16), 4))

    a_peramb = np.zeros((nt, N_dist, N_dist, 4, 4), dtype=np.cdouble)
    for n, m in it.product(range(N_dist), repeat=2):
        for s0, s1 in it.product(range(4), repeat=2):
            a_peramb[:,n,m,s0,s1] = d_peramb[f"output/{prec}/n_{n}_{m}_s_{s0}_{s1}_t_{t0}_ins_{ts}"]

    return a_peramb


def load_peramb(src, t0s, prec, N_dist=None):
    print(f"{qstatpy.timer.__str__()}Loading perambulators", flush=True)
    perambs = {t: load_single_peramb(f"{src}/pm_contr_{prec}_t{t}", N_dist) for t in t0s}
    return perambs 



def load_mins(fn, N_dist=None):
    print(f"{qstatpy.timer.__str__()}Loading momentum insertions", flush=True)
    d_mins = qstatpy.gpt_io.get_data(fn+'/head.dat')

    tag_struct = next(iter(d_mins))
    nt = d_mins[tag_struct].shape[0]
    mtags = list(set([tag.split('/')[0] for tag in d_mins.keys()]))
    if N_dist is None:
        N_dist = d_mins[tag_struct].shape[1]

    mvals = []
    for mtag in mtags:
        mvals.append(tuple([int(v) for v in mtag.split('_')[1:]]))

    a_mins = {pval: np.zeros((nt, N_dist, N_dist), dtype=np.cdouble) for pval in mvals}

    for p in mvals:
        for n, m in it.product(range(N_dist), repeat=2):
            a_mins[p][:,n,m] = d_mins[f"p_{p[0]}_{p[1]}_{p[2]}/n_{n}_{m}"]

    return a_mins

