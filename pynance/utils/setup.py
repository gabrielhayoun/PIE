import pynance
from datetime import datetime

def get_results_dir(name, add_date=False, create_dir=True):
    if(add_date):
        now = datetime.now().strftime(format='%Y%m%d%H%M')
        results_dir = pynance.utils.user.get_path_to_results() / '{}_{}'.format(now, name)
    else:
        results_dir = pynance.utils.user.get_path_to_results() / '{}'.format(name)
    
    if(create_dir):
        results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir

# --------------- setup ------------------- #
def setup_pipeliner_params(parameters, kind):
    framework = parameters['general']['framework']
    task = parameters['general']['task']

    preprocessor_class, preprocessor_parameters = pynance.utils.setup.setup_preprocessor_section(parameters['preprocessor'])
    preprocesser = preprocessor_class(**preprocessor_parameters)

    parameters['general'] = pynance.utils.setup.setup_general_section(parameters['general'])
    model_class, model_parameters = pynance.utils.setup.setup_model_section(parameters['model'])
    trainer_class, trainer_parameters = pynance.utils.setup.setup_training_section(parameters['training'], framework, task, parameters['general']['results_dir'])
    
    # TODO: should not be here I feel.
    if(kind == 'infer'):
        train_data = None
        test_data = None
    elif(kind=='train'):
        dict_stocks = pynance.utils.setup.setup_data_section(parameters['data'])
        
        preprocesser.init_preprocessor(dict_stocks)
        
        scaler_path = parameters['general']['results_dir'] / 'scaler'
        scaler_path.mkdir(parents=False, exist_ok=True)
        preprocesser.save(scaler_path)
        
        train_data = dict_stocks
        test_data = None

    dataloader_class, dataloader_parameters = pynance.utils.setup.setup_dataloader_section(parameters['dataloader'],
                                                                                           preprocesser,
                                                                                           framework,
                                                                                           task,
                                                                                           train_data,
                                                                                           test_data)
    # TODO: add analyze
    pipeliner_params = {
        'model_class': model_class,
        'dataloader_class': dataloader_class,
        'trainer_class': trainer_class,
        'analyzer_class': None,
        'model_parameters': model_parameters,
        'dataloader_parameters': dataloader_parameters,
        'trainer_parameters': trainer_parameters,
        'analyzer_parameters': None
    }
    return pipeliner_params

def setup_general_section(parameters):
    parameters['results_dir'] = pynance.utils.setup.get_results_dir(parameters['name'])
    return parameters

def setup_model_section(parameters):
    model_type = parameters['model_type']
    if(model_type == 'GRU'):
        model_class = pynance.model.forecasting.TFnaive
    if(model_type == 'MultipleLinearRegression'):
        model_class = pynance.model.regression.MultipleLinearRegression
    return model_class, parameters[model_type]

def setup_preprocessor_section(parameters):
    preprocesser = pynance.utils.preprocessing.Preprocesser
    return preprocesser, parameters

def setup_dataloader_section(parameters,
                             preprocessor,
                             framework,
                             task,
                             train_data,
                             test_data):
    dataloader_class = pynance.utils.conventions.get_dataloader_from_task(task)
    dataloader_type = parameters['dataloader_type']
    params = {}
    for key, value in parameters.items():
        if(key != 'dataloader_type'):
            try:
                value = dict(value)
                if(key==dataloader_type):
                    for key_, val in value.items():
                        params[key_] = val
            except:
                params[key] = value        
            
    params['preprocessor'] = preprocessor
    params['framework'] = framework
    params['train_data'] = train_data
    params['test_data'] = test_data
    return dataloader_class, params

def setup_training_section(parameters, framework, task, results_dir):
    trainer_class = pynance.utils.conventions.get_trainer(framework, task)
    trainer_parameters = {}
    if(framework == 'pytorch'):
        import torch
        trainer_parameters = parameters['pytorch']
        trainer_parameters['collater_fn'] = pynance.utils.datasets.collaters.TimeSeriesCollater(
            dtype=torch.float, device=torch.device(trainer_parameters['device']))
    else:
        trainer_parameters = parameters['sklearn']
    trainer_parameters['saving_dir'] = results_dir
    return trainer_class, trainer_parameters

def setup_data_section(parameters):
    tickers = pynance.data.readers.read_txt(parameters['tickers_file_name'])
    parameters['tickers'] = tickers
    dict_stocks = pynance.data.readers.get_financial_datas(
        tickers, start=parameters['start_date'], end=parameters['end_date'],
        conversion=True, return_type=parameters['return_type'])
    parameters['dict_stocks'] = dict_stocks
    for key, value in dict_stocks.items():
        print(f'{key} - rows:{len(value)} - columns: {value.columns}')
    return dict_stocks

def setup_inference_section(parameters):
    results_path = pynance.utils.user.get_path_to_results()
    parameters['prediction_model'] = results_path / parameters['prediction_model']
    parameters['regression_model'] = results_path / parameters['regression_model']
    return parameters
    

# -------------- Init pipeliner ------------------ #

def create_and_init_pipeliner(
        model_class,
        dataloader_class,
        trainer_class,
        analyzer_class,
        model_parameters,
        dataloader_parameters,
        trainer_parameters,
        analyzer_parameters):
    pipeliner = pynance.utils.pipeliners.Pipeliner(
        model_class,
        dataloader_class,
        trainer_class=trainer_class,
        analyser_class=analyzer_class
    )
    pipeliner.init_model(model_parameters)
    pipeliner.init_dataloader(dataloader_parameters)
    pipeliner.init_trainer(trainer_parameters)
    pipeliner.init_analyser(analyzer_parameters)
    return pipeliner

# def init_pipeliner(parameters, results_dir, inference=False):
#     framework = parameters['general']['framework']
#     task = parameters['general']['task']

#     dataloader_class = pynance.utils.conventions.get_dataloader_from_task(task)
#     trainer_class = pynance.utils.conventions.get_trainer(framework, task)
#     preprocesser = pynance.utils.preprocessing.Preprocesser(**parameters['preprocessor'])

#     dict_stocks = parameters['data']['dict_stocks']
#     preprocesser.init_preprocessor(dict_stocks)

#     scaler_path = results_dir / 'scaler'
#     scaler_path.mkdir(parents=False, exist_ok=True)
#     preprocesser.save(scaler_path)

#     if(framework == 'pytorch'):
#         import torch
#         parameters['training']['parameters'] = parameters['training']['pytorch']
#         parameters['training']['parameters']['collater_fn'] = pynance.utils.datasets.collaters.TimeSeriesCollater(
#             dtype=torch.float, device=torch.device(parameters['training']['parameters']['device']))

#     else:
#         parameters['training']['parameters'] = parameters['training']['sklearn']
#     trainer_params = parameters['training']['parameters']
#     trainer_params['saving_dir'] = parameters['general']['results_dir']

#     np.random.seed(parameters['general']['seed'])

#     # Pipeliner
#     pipeliner = pynance.utils.pipeliners.Pipeliner(
#         model_class=parameters['model']['model_class'],
#         dataloader_class=dataloader_class,
#         trainer_class=trainer_class,
#         analyser_class=None
#     )

#     pipeliner.init_model(parameters['model']['parameters'])
#     if(inference):
#         train_data = None
#         test_data = dict_stocks
#     else:
#         train_data = dict_stocks
#         test_data = None
#     pipeliner.init_dataloader(train_data, test_data, preprocesser, framework, parameters['dataloader'])
#     pipeliner.init_trainer(trainer_params)
#     pipeliner.init_analyser({})
#     return pipeliner




