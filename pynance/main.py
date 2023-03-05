import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import pynance
from datetime import datetime

def main(path_to_cfg, save=True):
    p = pynance.config.cfg_reader.read(path_to_cfg) # parameters
    # setup(p) # modify p in place
    print(p)
    
    # if(not os.path.isdir(p['setup']['path'])):
    #     os.makedirs(p['setup']['path']) 

    # if(save): # saving the new params to a file so the user can go and debug it or refer to all the simulations params when needed.
    #     from configobj import ConfigObj
    #     pp_dict = convert_objects(p.dict())
    #     pp = ConfigObj(pp_dict)
    #     pp.filename = '{}/{}.ini'.format(p['setup']['path'], 'params')
    #     pp.write()
        
    # dealing with seeding
    # np.random.seed(p['simulation']['seed'])

    # TODO :
    # - add plot functions and params
    # - add more complexe system (with parts of the system that we dont take) - example of the cylinder.
    # - maybe some verbose and saving of the params 
    
    return True

# ----------------- convert ------------------- #
def convert_objects(p):
    pp = {}
    for k, v in p.items():
        if(type(v) is dict):
            pp[k] = convert_objects(v)
        elif(type(v) is list):
            for i in range(len(v)):
                v[i] = convert_object(v[i])
            pp[k] = v
        else:
            pp[k] = convert_object(v)
    return pp

def convert_object(o):
    try :
        return '{}{}'.format(o.__name__, inspect.signature(o))
    except Exception:
        try :
            return o.__str__()
        except Exception:
            return o
