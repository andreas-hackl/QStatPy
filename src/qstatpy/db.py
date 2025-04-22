import os
import numpy as np
import qstatpy.np_json as json

class Database:
    def __init__(self, file):
        self.file = file
        if os.path.isfile(self.file):
            with open(self.file) as fp:
                self.database = json.load(fp)
        else:
            self.database = {}

    def content(self, verbosity=0):
        s = "DATABASE CONSISTS OF\n"
        for tag, tag_dict in self.database.items():
            s += f"{tag:20s}\n"
            if verbosity >= 1:
                for sample_tag, sample_dict in tag_dict.items():
                    s += f'└── {sample_tag:20s}\n'
                    if verbosity >= 2:
                        for cfg_tag, val in sample_dict.items():
                            s += f'\t└── {cfg_tag}\n'
                            if verbosity >= 3:
                                s += '\t\t' + f'{val.__str__()}'.replace('\n', '\n\t\t')
                            s += '\n'
        s += '\n'
        return s

    def print(self, verbosity=0):
        print(self.content(verbosity=verbosity))

    def __str__(self):
        return self.content(verbosity=0) 

    def add_data(self, data, tag, sample_tag, cfg_tag=None):

        if cfg_tag is None and not isinstance(data, dict):
            raise ValueError()

        if tag in self.database:
            if sample_tag in self.database[tag]:
                if isinstance(data, dict):
                    for cfg_tag, cfg_data in data.items():
                        self.database[tag][sample_tag][cfg_tag] = cfg_data
                else:
                    self.database[tag][sample_tag][cfg_tag] = data
            else:
                if isinstance(data, dict):
                    self.database[tag][sample_tag] = data
                else:
                    self.database[tag][sample_tag] = {cfg_tag: data}
        else:
            if isinstance(data, dict):
                self.database[tag] = {sample_tag: data}
            else:
                self.database[tag] = {sample_tag: {cfg_tag: data}}

        
    def get_data(self, tag, sample_tag):
        if sample_tag in self.database[tag]:
            return dict(self.database[tag][sample_tag])
        return None

    
    # STATISTICAL FUNCTIONS

    def jackknife(self, load_tag, store_tag, f, eps=1.0, **fargs):
        data = self.get_data(*load_tag)
        tag, jk_tag = store_tag
        N = len(data)
        mean = np.mean(np.array(list(data.values())), axis=0)
        if jk_tag not in self.database[tag]:
            self.database[tag][jk_tag] = {}

        self.database[tag][jk_tag]['mean'] = mean
        for cfg_tag in data.keys():
            self.database[tag][jk_tag][cfg_tag] = f( mean + eps * (mean - data[cfg_tag]) / (N - 1), **fargs)

    def estimate(self, load_tag, store_tag=None, f=lambda x: x, eps=1.0, fmean=None, **fargs):
        jk_data = self.get_data(*load_tag)
        
        N = len(jk_data) - 1 # -1 since mean is part of jk_data

        if store_tag is None:
            tag = None
        else:
            tag, jk_tag = store_tag

        mean = jk_data['mean']

        if fmean is None:
            fmean = f(mean, **fargs)
        if not tag is None:
            self.add_data(fmean, tag, jk_tag, 'mean')

        Ndim = len(fmean)
        fcov = np.zeros((Ndim, Ndim), dtype=fmean.dtype)

        for cfg_tag, di in jk_data.items():
            if cfg_tag != 'mean':

                fi = f(di, **fargs)

                if not tag is None:
                    self.add_data(fi, tag, jk_tag, cfg_tag)

                yi = fi - fmean 
                fcov += np.outer(np.conjugate(yi), yi)

        fcov *= (N-1)/(N*eps**2)

        return fcov

    def mean(self, tag, sample_tag):
        return self.database[tag][sample_tag]['mean']

    def cov(self, tag, sample_tag, eps=1.0):
        return self.estimate([tag, sample_tag], eps=eps)

    def std(self, tag, sample_tag, eps=1.0):
        return np.sqrt(np.diag(self.cov(tag, sample_tag, eps=eps)))

    def curve(self, tag, sample_tag, eps=1.0):
        y = self.mean(tag, sample_tag)
        ys = self.std(tag, sample_tag, eps=eps)
        x = np.arange(len(y))
        return x, y, ys

    def mcurve(self, tag, sample_tag, xrange, eps=1.0):
        x, y, ys = self.curve(tag, sample_tag, eps=1.0)
        m = np.zeros_like(x, dtype=bool)
        m[xrange[0]:xrange[1]+1] = True
        return x[m], y[m], ys[m]

    def __call__(self, tag, sample_tag):
        return self.get_data(tag, sample_tag)


    def combine(self, dst_tag, tags, f, allow_mean_filling=False):
        d = {tag: self.get_data(*tag) for tag in tags}
        cfg_tag_dict = {tag: list(d[tag].keys()) for tag in tags}
        dmeans = {tag: self.mean(*tag) for tag in tags}

        if allow_mean_filling:
            cfg_tags = list(set(sum([list(di.keys()) for di in d.values()], [])))
        else:
            cfg_tags = list(set.intersection(*map(set,list(cfg_tag_dict.values()))))

        nd = {}
        for cfg_tag in cfg_tags:
            v = {tag: d[tag].get(cfg_tag, dmeans[tag]) for tag in tags}
            nd[cfg_tag] = f(v)

        self.add_data(nd, *dst_tag)


    def operation(self, in_tag, f, out_tag=None):
        d = self.get_data(*in_tag)
        nd = {}
        for cfg_tag, v in d.items():
            nd[cfg_tag] = f(v)

        if not out_tag is None:
            self.add_data(nd, out_tag[0], out_tag[1])
        else:
            return nd
        

    def save(self):
        with open(self.file, "w") as f:
            json.dump(self.database, f)


    def __eq__(self, other):
        return self.database == other.database


    def remove(self, tag, sample_tag=None):
        if self.database.get(tag, None) is None:
            return

        if sample_tag is None:
            self.database.pop(tag)
        else:
            if self.database[tag].get(sample_tag, None) is None:
                return 
            self.database[tag].pop(sample_tag)
    
