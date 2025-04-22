import qstatpy.np_json
import qstatpy.db
import qstatpy.io
import qstatpy.gevp 
import qstatpy.gpt_io
import qstatpy.distillation
import qstatpy.fit


from qstatpy.db import Database
from qstatpy.io import load, get_tags

import time

class Timer():
    def __init__(self):
        self.t_begin = time.time()

    def __call__(self):
        return time.time() - self.t_begin
    
    def __str__(self):
        return f"LQCDPY  {self.__call__():20.5f} sec  :   "
    
timer = Timer()


def message(s, *args):
    print(f"{timer.__str__()}"+s, *args, flush=True)
