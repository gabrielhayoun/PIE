import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import pynance
from datetime import datetime

def main(path_to_cfg, save=True):
    parameters = pynance.config.cfg_reader.read(path_to_cfg) # parameters
    setup(parameters)

    if(not os.path.isdir(parameters['general']['results_path'])):
        os.makedirs(parameters['general']['results_path']) 

    if(save): # saving the new params to a file so the user can go and debug it or refer to all the simulations params when needed.
        from configobj import ConfigObj
        pp_dict = convert_objects(parameters.dict())
        pp = ConfigObj(pp_dict)
        pp.filename = '{}/{}.ini'.format(parameters['general']['results_path'], 'parameters')
        pp.write()
        
    np.random.seed(parameters['general']['seed'])
    # TODO: add set pytorch seeding

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

# --------------- setup ------------------- #

def setup(parameters):
    # objective : converts to write format and do everything required on the set of parameters for later use
    parameters['general'] = general_setup(parameters['general'])
    parameters['model'] = model_setup(parameters['model'])
    parameters['data'] = data_setup(parameters['data'])
    parameters['training'] = training_setup(parameters['training'])
    return parameters

def general_setup(parameters):
    now = datetime.now().strftime(format='%Y%m%d%H%M')
    parameters['results_path'] = pynance.utils.user.get_path_to_results() / '{}_{}'.format(now, parameters['name'])
    return parameters

def model_setup(parameters):
    return parameters

def data_setup(parameters):
    return parameters

def training_setup(parameters):
    return parameters




