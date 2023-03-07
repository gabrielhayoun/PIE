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

def setup_general_section(parameters):
    parameters['results_dir'] = pynance.utils.setup.get_results_dir(parameters['name'])
    return parameters

def setup_model_section(parameters):
    model_type = parameters['model_type']
    if(model_type == 'GRU'):
        parameters['model_class'] = pynance.model.forecasting.TFnaive
    if(model_type == 'MultipleLinearRegression'):
        parameters['model_class'] = pynance.model.regression.MultipleLinearRegression
    parameters['parameters'] = parameters[model_type]
    return parameters

# def setup_data_section(parameters, task, framework):
#     start_date = parameters['start_date']
#     end_date = parameters['end_date']
#     index_ticker = parameters['index_ticker']
#     return_type = parameters['return_type']
#     preprocesser = pynance.utils.preprocessing.Preprocesser(**parameters['preprocessing'])

#     if(task == 'regression'):
#         tickers = pynance.data.readers.read_txt(parameters['regression']['tickers_file_name'])
#         parameters['tickers'] = tickers
#         dict_stocks = pynance.data.readers.get_financial_datas(tickers + [index_ticker], start=start_date, end=end_date, conversion=True, return_type=return_type)
#         preprocesser.init_preprocessor(dict_stocks)
#         parameters['parameters'] = {
#             'train_data': dict_stocks,
#             'test_data': None,
#             'framework': framework,
#             'ratio': parameters['ratio'],
#             'index': len(tickers), # index of index_ticker in the list of data
#             'preprocesser': preprocesser
#         }
#     elif(task == 'prediction'):
#         dict_stocks = pynance.data.readers.get_financial_datas([index_ticker], start=start_date, end=end_date, conversion=True, return_type=return_type)
#         preprocesser.init_preprocessor(dict_stocks)
#         parameters['parameters'] = {
#             'train_data': dict_stocks,
#             'test_data': None,
#             'framework': framework,
#             'ratio': parameters['ratio'],
#             'window': parameters['prediction']['window'],
#             'preprocesser': preprocesser
#         }
#     return parameters

def setup_preprocessor_section(parameters):
    return parameters

def setup_dataloader_section(parameters):
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
    return params

def setup_training_section(parameters):
    # if(framework == 'pytorch'):
    #     parameters['parameters'] = parameters['pytorch']
    # elif(framework == 'sklearn'):
    #     parameters['parameters'] = parameters['sklearn']
    return parameters

def setup_data_section(parameters):
    tickers = pynance.data.readers.read_txt(parameters['tickers_file_name'])
    parameters['tickers'] = tickers
    dict_stocks = pynance.data.readers.get_financial_datas(
        tickers, start=parameters['start_date'], end=parameters['end_date'],
        conversion=True, return_type=parameters['return_type'])
    parameters['dict_stocks'] = dict_stocks
    for key, value in dict_stocks.items():
        print(f'{key} - rows:{len(value)} - columns: {value.columns}')
    return parameters

def setup_inference_section(parameters):
    return parameters
    # prediction_model = parameters['prediction_model']
    # regression_model = parameters['regression_model']
    # window = parameters['window']
    
    # # loading prediction model

    # loading regression model