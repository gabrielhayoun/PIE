import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import pynance
from datetime import datetime
import torch
from functools import partial

#TODO: make it better. For now it's stupid because values are hardcoded.

def main(path_to_cfg, save=True):
    parameters = pynance.config.cfg_reader.read(path_to_cfg) # parameters
    setup(parameters)

    if(not os.path.isdir(parameters['general']['results_dir'])):
        os.makedirs(parameters['general']['results_dir']) 

    if(save): # saving the new params to a file so the user can go and debug it or refer to all the simulations params when needed.
        from configobj import ConfigObj
        pp_dict = convert_objects(parameters.dict())
        pp = ConfigObj(pp_dict)
        pp.filename = '{}/{}.ini'.format(parameters['general']['results_dir'], 'parameters')
        pp.write()

    framework = parameters['general']['framework']
    task = parameters['general']['task']
    dataloader_class = pynance.utils.conventions.get_dataloader_from_task(task)
    trainer_class = pynance.utils.conventions.get_trainer(framework, task)

    np.random.seed(parameters['general']['seed'])

    pipeliner = pynance.utils.pipeliners.Pipeliner(
        model_class=parameters['model']['model_class'],
        dataloader_class=dataloader_class,
        trainer_class=trainer_class,
        analyser_class=None
    )
    pipeliner.init_model(parameters['model']['parameters'])
    pipeliner.init_dataloader(parameters['data']['parameters'])
    trainer_params = parameters['training']['parameters']
    if(framework == 'pytorch'):
        import torch
        trainer_params['collater_fn'] = pynance.utils.datasets.collaters.TimeSeriesCollater(dtype=torch.float, device=torch.device(trainer_params['device']))
    trainer_params['saving_dir'] = parameters['general']['results_dir']
    pipeliner.init_trainer(trainer_params)
    pipeliner.init_analyser({})

    pipeliner.train()
    pipeliner.evaluate()
    pipeliner.analyze()

    return True

# --------------- setup ------------------- #
def setup(parameters):
    framework = parameters['general']['framework']
    task = parameters['general']['task']
    # objective : converts to right format and do everything required on the set of parameters for later use
    parameters['general'] = general_setup(parameters['general'])
    parameters['data'] = data_setup(parameters['data'], task, framework)
    parameters['model'] = model_setup(parameters['model'])
    parameters['training'] = training_setup(parameters['training'], framework)
    return parameters

def general_setup(parameters):
    now = datetime.now().strftime(format='%Y%m%d%H%M')
    parameters['results_dir'] = pynance.utils.user.get_path_to_results() / '{}_{}'.format(now, parameters['name'])
    return parameters

def model_setup(parameters):
    model_type = parameters['model_type']
    if(model_type == 'GRU'):
        parameters['model_class'] = pynance.model.forecasting.TFnaive
    if(model_type == 'MultipleLinearRegression'):
        parameters['model_class'] = pynance.model.regression.MultipleLinearRegression
    parameters['parameters'] = parameters[model_type]
    return parameters

def data_setup(parameters, task, framework):
    start_date = parameters['start_date']
    end_date = parameters['end_date']
    index_ticker = parameters['index_ticker']
    return_type = parameters['return_type']
    preprocesser = pynance.utils.preprocessing.Preprocesser(**parameters['preprocessing'])

    if(task == 'regression'):
        tickers = pynance.data.readers.read_txt(parameters['regression']['tickers_file_name'])
        parameters['tickers'] = tickers
        dict_stocks = pynance.data.readers.get_financial_datas(tickers + [index_ticker], start=start_date, end=end_date, conversion=True, return_type=return_type)
        preprocesser.init_preprocessor(dict_stocks)
        parameters['parameters'] = {
            'train_data': dict_stocks,
            'test_data': None,
            'framework': framework,
            'ratio': parameters['ratio'],
            'index': len(tickers), # index of index_ticker in the list of data
            'preprocesser': preprocesser
        }
    elif(task == 'prediction'):
        dict_stocks = pynance.data.readers.get_financial_datas([index_ticker], start=start_date, end=end_date, conversion=True, return_type=return_type)
        preprocesser.init_preprocessor(dict_stocks)
        parameters['parameters'] = {
            'train_data': dict_stocks,
            'test_data': None,
            'framework': framework,
            'ratio': parameters['ratio'],
            'window': parameters['prediction']['window'],
            'preprocesser': preprocesser
        }
    return parameters

def training_setup(parameters, framework):
    if(framework == 'pytorch'):
        parameters['parameters'] = parameters['pytorch']
    elif(framework == 'sklearn'):
        parameters['parameters'] = parameters['sklearn']
    return parameters


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
