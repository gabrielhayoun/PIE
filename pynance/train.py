import numpy as np

# local
import pynance

#TODO: make it better. For now it's stupid because values are hardcoded.
# Idea: I should be able to call methods here to returned everything that is necessary for inference (amongst other)
# with models with trained params

def main(path_to_cfg):
    # General parameters
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='train')
    results_dir = pynance.utils.setup.get_results_dir(parameters['general']['name'])
    pynance.utils.saving.save_configobj(parameters, results_dir, 'parameters') 

    # Pipeliner parameters: load data, create model / dataloader / trainer class etc.
    pipeliner_params = pynance.utils.setup.setup_pipeliner_params(parameters, kind='train')
    pynance.utils.saving.save_configobj(pipeliner_params, results_dir, 'pipeliner_params')

    pipeliner = pynance.utils.setup.create_and_init_pipeliner(**pipeliner_params)
    
    pipeliner.train()
    # pipeliner.evaluate()
    # pipeliner.analyze()

    return True
