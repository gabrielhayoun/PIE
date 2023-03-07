import numpy as np

# local
import pynance

#TODO: make it better. For now it's stupid because values are hardcoded.
# Idea: I should be able to call methods here to returned everything that is necessary for inference (amongst other)
# with models with trained params

def main(path_to_cfg):
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='train') # 'raw' parameters
    results_dir = pynance.utils.setup.get_results_dir(parameters['general']['name'])

    pynance.utils.saving.save_configobj(parameters, results_dir, 'parameters') # for later use

    # Modify params in place
    setup(parameters)

    pynance.utils.saving.save_configobj(parameters, results_dir, 'processed_parameters')

    framework = parameters['general']['framework']
    task = parameters['general']['task']

    dataloader_class = pynance.utils.conventions.get_dataloader_from_task(task)
    trainer_class = pynance.utils.conventions.get_trainer(framework, task)
    preprocesser = pynance.utils.preprocessing.Preprocesser(**parameters['preprocessor'])

    dict_stocks = parameters['data']['dict_stocks']
    preprocesser.init_preprocessor(dict_stocks)

    if(framework == 'pytorch'):
        import torch
        parameters['training']['parameters'] = parameters['training']['pytorch']
        parameters['training']['parameters']['collater_fn'] = pynance.utils.datasets.collaters.TimeSeriesCollater(
            dtype=torch.float, device=torch.device(parameters['training']['parameters']['device']))

    else:
        parameters['training']['parameters'] = parameters['training']['sklearn']
    trainer_params = parameters['training']['parameters']
    trainer_params['saving_dir'] = parameters['general']['results_dir']

    np.random.seed(parameters['general']['seed'])

    # Pipeliner
    pipeliner = pynance.utils.pipeliners.Pipeliner(
        model_class=parameters['model']['model_class'],
        dataloader_class=dataloader_class,
        trainer_class=trainer_class,
        analyser_class=None
    )

    pipeliner.init_model(parameters['model']['parameters'])
    pipeliner.init_dataloader(dict_stocks, None, preprocesser, framework, parameters['dataloader'])
    pipeliner.init_trainer(trainer_params)
    pipeliner.init_analyser({})

    pipeliner.train()
    # pipeliner.evaluate()
    # pipeliner.analyze()

    return True

# --------------- setup ------------------- #
def setup(parameters):
    framework = parameters['general']['framework']
    task = parameters['general']['task']
    # objective : converts to right format and do everything required on the set of parameters for later use
    parameters['general'] = pynance.utils.setup.setup_general_section(parameters['general'])
    parameters['data'] = pynance.utils.setup.setup_data_section(parameters['data'])
    parameters['model'] = pynance.utils.setup.setup_model_section(parameters['model'])
    parameters['training'] = pynance.utils.setup.setup_training_section(parameters['training'])
    parameters['preprocessor'] = pynance.utils.setup.setup_preprocessor_section(parameters['preprocessor'])
    parameters['dataloader'] = pynance.utils.setup.setup_dataloader_section(parameters['dataloader'])
    return parameters


