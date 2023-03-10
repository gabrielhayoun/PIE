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
    # General parameters
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='infer')
    results_dir = pynance.utils.setup.get_results_dir(parameters['general']['name'])
    pynance.utils.saving.save_configobj(parameters, results_dir, 'parameters')
     
    (pipeliner_pred, pred_parameters), (pipeliner_regr, regr_parameters) = fetch_inference_model(parameters)
    data_dicts, index_name = check_coherence_and_return_data(pred_parameters['data'], regr_parameters['data'])

    index_dict = {index_name: data_dicts[index_name]}

    pred_dict = pipeliner_pred.predict(index_dict, {'window': 10})
    pred_dict_regr = pipeliner_regr.predict(pred_dict, {})

    df_coint = pynance.coint.load_coint_file(parameters['general']['coint_name'])

    for key, df_ in pred_dict_regr.items():
        pred_dict_regr[key] = df_[parameters['strategy']['feature']].values
    
    pynance.strategy.basic.get_best_action(
            df_coint,
            pred_dict_regr,
            **parameters['strategy']['best_action']
    )
    # TODO: print and save
    
    return True

def fetch_inference_model(parameters):
    prediction_model = parameters['inference']['prediction_model']
    pred_root_dir_results = (pynance.utils.user.get_path_to_results() / prediction_model)

    regression_model = parameters['inference']['regression_model']
    regr_root_dir_results = (pynance.utils.user.get_path_to_results() / regression_model)

    pred_parameters = get_parameters(parameters, pred_root_dir_results)
    regr_parameters = get_parameters(parameters, regr_root_dir_results)
    
    pipeliner_pred = get_pipeliner(pred_parameters)
    pred_model_dir = pred_root_dir_results / 'model'
    pred_scaler_dir = pred_root_dir_results / 'scaler'
    
    pipeliner_pred.load_model_state(pred_model_dir)
    pipeliner_pred.load_scaler(pred_scaler_dir)

    pipeliner_regr = get_pipeliner(regr_parameters)
    regr_model_dir = regr_root_dir_results / 'model'
    regr_scaler_dir = regr_root_dir_results / 'scaler'

    pipeliner_regr.load_model_state(regr_model_dir)
    pipeliner_regr.load_scaler(regr_scaler_dir)

    return (pipeliner_pred, pred_parameters), (pipeliner_regr, regr_parameters)

def get_parameters(influence_parameters, root_dir_results):
    print(f'Loading model from {root_dir_results}')
    path_to_cfg =  root_dir_results/ 'parameters.cfg'
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='train')
    parameters = replace_parameters_for_inference(influence_parameters, parameters)
    return parameters

def get_pipeliner(parameters):
    pipeliner_params = pynance.utils.setup.setup_pipeliner_params(parameters, kind='infer')
    pipeliner = pynance.utils.setup.create_and_init_pipeliner(**pipeliner_params)
    return pipeliner

def replace_parameters_for_inference(infer_parameters, train_parameters):
    # for now
    train_parameters['general']['name'] = infer_parameters['general']['name']
    train_parameters['data']['start_date'] = get_start_date(
        infer_parameters['inference']['start_prediction_date'],
        - infer_parameters['inference']['training_window']
        )
    train_parameters['data']['end_date'] = infer_parameters['inference']['start_prediction_date']
    return train_parameters

def make_dates(length_preds, init_date=None, end_date=None):
    assert(init_date is not None or end_date is not None)
    import datetime
    dates = []
    date = init_date
    dt = datetime.timedelta(days=1)
    if(init_date is None):
        dt = -dt
        date = end_date
    while(len(dates) < length_preds):
        date += dt
        if(date.isoweekday() <= 5):
            dates.append(date)
    if(init_date is None):
        return dates[::-1]
    return dates

def get_start_date(date, delta):
    from datetime import datetime, timedelta
    format = '%Y-%m-%d'
    date = datetime.strptime(date, format)
    list_dates = make_dates(-delta, None, date)
    first_date = list_dates[0].strftime(format=format)
    return first_date

def check_coherence_and_return_data(pred_data, regr_data):
    dict_stocks_pred = pynance.utils.setup.setup_data_section(pred_data) 
    dict_stocks_regr = pynance.utils.setup.setup_data_section(regr_data)
    assert(len(dict_stocks_pred) == 1)
    index_name = list(dict_stocks_pred.keys())[0]
    assert(index_name in dict_stocks_regr.keys())
    return dict_stocks_regr, index_name