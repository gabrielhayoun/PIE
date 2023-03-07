import numpy as np
from datetime import datetime

# local
import pynance

""" 
1) Read params from config files
2) get models path from params
3) Load prediction model and regression models
4) Make prediction
5) Possibly call strategy here ?
"""

def main(path_to_cfg):
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='infer') # 'raw' parameters
    results_dir = pynance.utils.setup.get_results_dir(parameters['general']['name'])

    pynance.utils.saving.save_configobj(parameters, results_dir, 'parameters') # for later use

    # Modify params in place
    setup(parameters)
    pynance.utils.saving.save_configobj(parameters, results_dir, 'processed_parameters')


def setup(parameters):
    pass