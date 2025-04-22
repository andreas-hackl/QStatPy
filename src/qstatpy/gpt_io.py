import sys, os, struct, binascii
from fnmatch import fnmatch
import numpy as np
import itertools as it
# code similar to https://github.com/lehner/gpt/blob/master/lib/gpt/core/io/corr_io.py

R_NONE    = 0x00
R_EMPTY   = 0x01
R_REAL    = 0x02
R_IMAG    = 0x04
R_SYMM    = 0x08
R_ASYMM   = 0x10

def flag_str(f):
    r=""
    if f & R_EMPTY:
        r += "empty "
    if f & R_REAL:
        r += "real "
    if f & R_IMAG:
        r += "imag "
    if f & R_SYMM:
        r += "symm "
    if f & R_ASYMM:
        r += "asymm "
    return r.strip()


def get_tags(fn):
    tags = []
    f = open(fn, 'rb')
    try:
        while True:
            rd = f.read(4)
            if len(rd) == 0:
                break

            ntag = struct.unpack('i', rd)[0]
            tag = f.read(ntag)
            (crc32, ln, flags) = struct.unpack('IHH', f.read(4*2))
            tags.append(tag[0:-1].decode('utf8'))

            nf = 1
            if flags & (R_REAL|R_IMAG):
                nf = 2
            if flags & (R_SYMM|R_ASYMM):
                ln = ln/2+1
            if flags & R_EMPTY:
                ln = 0

            f.seek(ln*16 // nf, 1)
        f.close()
    except:
        raise
    return tags


class gdict(dict):
    def glob(self, match):
        return dict([(k,v) for k, v in self.items() if fnmatch(k, match)])

    def glob_mean(self, match):
        data = self.glob(match)
        return 1/len(data) * sum([v for v in data.values()])

    def glob_sum(self, match):
        data = self.glob(match)
        return sum([v for v in data.values()])




def reconstruct_full(flags, i):
    if flags & R_SYMM:
        N=len(i)/2
        for j in range(N/2+1,N):
            jm=N - j
            i[2*j + 0] = i[2*jm + 0]
            i[2*j + 1] = -i[2*jm + 1]

    if flags & R_ASYMM:
        N=len(i)/2
        for j in range(N/2+1,N):
            jm=N - j
            i[2*j + 0] = -i[2*jm + 0]
            i[2*j + 1] = i[2*jm + 1]

def reconstruct_min(flags, i, NT):
    if flags & R_EMPTY:
        return [ 0.0 for l in 2*range(NT) ]
    if flags == 0:
        return i

    o=[ 0.0 for l in 2*range(NT) ]

    # first fill in data at right places
    i0=0
    istep=1

    if flags & R_REAL:
        istep=2
    if flags & R_IMAG:
        istep=2
        i0=1

    for j in range(len(i)):
        o[istep*j + i0] = i[j]

    return o


def get_data(fn):
    f=open(fn,"rb")
    data = gdict({})
    try:
        while True:
            rd=f.read(4)
            if len(rd) == 0:
                break
            ntag=struct.unpack('i', rd)[0]
            tag=f.read(ntag)
            (crc32,ln,flags)=struct.unpack('IHH', f.read(4*2))
    
            nf = 1
            lnr = ln
            if flags & (R_REAL|R_IMAG):
                nf = 2
            if flags & (R_SYMM|R_ASYMM):
                lnr = ln/2+1
            if flags & R_EMPTY:
                lnr = 0
    
            
            rd=reconstruct_min( flags, struct.unpack('d'*(2*lnr // nf), f.read(16*lnr // nf) ) , ln)
            crc32comp= ( binascii.crc32(struct.pack('d'*2*ln,*rd)) & 0xffffffff)

            reconstruct_full( flags, rd )
            
            if crc32comp != crc32:
                print("Data corrupted!")
                f.close()
                sys.exit(1)

            if flags != R_EMPTY:
                corr = []
                for j in range(ln):
                    corr.append(rd[j*2+0] + 1j*rd[j*2+1])
                corr = np.array(corr)
                data[tag[0:-1].decode('ascii')] = corr
    
            else:
                f.seek(lnr*16 / nf,1)
        f.close()
    except:
        raise

    return data

def get_types(tags, idx):
    return sorted(list(set([tag.split('/')[idx] for tag in tags])))


def find_all(tag, c):
    idx = tag.find(c)
    res = [idx]
    while idx != -1:
        idx = tag.find(c, idx+1)
        res.append(idx)
    return res[:-1]

def find_spliter(tag):
    i0s = find_all(tag, '-')
    i1s = find_all(tag, '.')
    spliter_pos = []
    for i0 in i0s:
        if i0-1 not in i1s:
            spliter_pos.append(i0)
    if len(spliter_pos) != 1:
        raise IndexError(f'Multiple spliter in {tag}')
    return spliter_pos[0]

def get_operator(tag):
    sidx = find_spliter(tag)
    return tag[:sidx], tag[sidx+1:]



class writer:
    def __init__(self, fn):
        self.f = open(fn, "w+b")

    def write(self, t, cc):
        tag_data = (t + "\0").encode("utf-8")
        self.f.write(struct.pack("i", len(tag_data)))
        self.f.write(tag_data)

        ln = len(cc)
        ccr = [fff for sublist in ((c.real, c.imag) for c in cc) for fff in sublist]
        bindata = struct.pack("d" * 2 * ln, *ccr)
        crc32comp = binascii.crc32(bindata) & 0xFFFFFFFF
        self.f.write(struct.pack("II", crc32comp, ln))
        self.f.write(bindata)
        self.f.flush()

    def write_spin(self, t, c):
        for mu, nu in it.product(range(4), repeat=2):
            self.write(f"{t}/s_{mu}_{nu}", c[:,mu,nu])


    def close(self):
        self.f.close()
