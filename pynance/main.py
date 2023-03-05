import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import pynance
from datetime import datetime
import torch

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
    parameters['data'] = data_setup(parameters['data'])
    parameters['model'] = model_setup(parameters['model'])
    parameters['training'] = training_setup(parameters['training'])
    return parameters

def general_setup(parameters):
    now = datetime.now().strftime(format='%Y%m%d%H%M')
    parameters['results_path'] = pynance.utils.user.get_path_to_results() / '{}_{}'.format(now, parameters['name'])
    return parameters

def model_setup(parameters):
    # should know how to load models
    model_type = parameters['stock_prediction_model_type']
    if(model_type == 'GRU'):
        gru_params = parameters['GRU']
        parameters['active_model'] = pynance.model.forecasting.TFnaive(
            input_size=gru_params['input_size'],
            hidden_size=gru_params['hidden_size'],
            num_layers=gru_params['num_layers']
        ) # .to(device=device, dtype=dtype)
    return parameters

def data_setup(parameters):
    parameters['actions'] = pynance.data.readers.read_txt(parameters['actions_file_name'])
    index = parameters['index']
    start_date = parameters['start_date']
    end_date = parameters['end_date']
    dict_stocks = pynance.data.readers.get_financial_datas(parameters['actions'] + [index], start=start_date, end=end_date, conversion=True, return_type='raw')
    parameters['df_actions'] = pd.DataFrame({stock: df_[pynance.utils.conventions.close_name] for stock, df_ in dict_stocks.items()}) 
    parameters['df_index'] = dict_stocks[index]
    return parameters

def training_setup(parameters):
    parameters['device'] = torch.device(parameters['device'])
    # parameters['dtype'] = torch.detype(parameters['dtype'])
    return parameters




