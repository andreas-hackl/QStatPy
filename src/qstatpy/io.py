import qstatpy
import os

def get_tags(path, configs):
    
    if not isinstance(configs, list):
        configs = [configs]

    tag_dict = {}
    for cfg in configs:
        data = qstatpy.np_json.load(open(f"{path}/{cfg}", 'r'))
        tags = list(data.keys())

        for tag in tags:
            if tag in tag_dict:
                tag_dict[tag].append(cfg)
            else:
                tag_dict[tag] = [cfg]

    return tag_dict


def load(db, path, configs, exact_tags, sloppy_tags=None, eps=1):
    """
        db                    Database
        path                  Path of the raw data
        configs               configs to consider; have to have the same name as the files in the path folder
        exact_tags            tags of correlator with exact and sloppy data or if sloppy_tags is None all tags
                              has the structure
                                  {tag0: [exact_tag00, exact_tag01], tag1: [exact_tag01, exact_tag01]}
        sloppy_tags           if None: only exact tags are consider, else where the only sloppy data is stored
                              same structure as exact_tags, also contains exact tags, where exact is replaced by sloppy
    """
    
    is_ama = sloppy_tags != None
    
    for cfg in configs:
        data = qstatpy.np_json.load(open(f"{path}/{cfg}", 'r'))
        
        for tag, etags in exact_tags.items():
            c_exact = 0
            c_sloppy = 0
            for etag in etags:
                c_exact += data[etag]
                if is_ama:
                    c_sloppy += data[etag.replace('exact', 'sloppy')]
                    
            c_exact /= len(etags)
            c_sloppy /= len(etags)
            
            db.add_data(c_exact, tag, 'EXACT_REF', cfg)
            if is_ama:
                db.add_data(c_sloppy, tag, 'SLOPPY_REF', cfg)
                
                db.add_data(c_exact - c_sloppy, tag, 'EPS_REF', cfg)
            
        if sloppy_tags is not None:
            for tag, stags in sloppy_tags.items():
                c_sloppy = 0
                for stag in stags:
                    c_sloppy += data[stag]
                    
                c_sloppy /= len(stags)
                    
                db.add_data(c_sloppy, tag, 'SLOPPY', cfg)
                
    funcs = {
        '': lambda x: x,
        '_real': lambda x: x.real,
        '_imag': lambda x: x.imag,
    }
    if is_ama:
        jks_tags = [('EXACT_REF', 'jk_exact_ref'),
                    ('SLOPPY_REF', 'jk_sloppy_ref'),
                    ('SLOPPY', 'jk_sloppy'),
                    ('EPS_REF', 'jk_eps')]

    else:
        jks_tags = [('EXACT_REF', 'jk_mean')]
        
    for tag in exact_tags.keys():

        for itag, otag in jks_tags:

            for ri_label, f in funcs.items():
                db.jackknife((tag, itag), (tag, f'{otag}{ri_label}'), f, eps=eps)
                
        if is_ama:
            for ri_label in funcs.keys():
                slp_tag = (tag, f'jk_sloppy{ri_label}')
                eps_tag = (tag, f'jk_eps{ri_label}')
                db.combine((tag, f'jk_mean{ri_label}'), [slp_tag, eps_tag], lambda v: v[slp_tag]+v[eps_tag])



class writer:

    def __init__(self, fn):
        self.d = dict()
        self.fn = fn
        if os.path.exists(self.fn):
            raise ValueError()

    def write(self, tag, c):
        self.d[tag] = c


    def close(self):
        with open(self.fn, 'w') as fp:
            qstatpy.np_json.dump(self.d, fp)
        
