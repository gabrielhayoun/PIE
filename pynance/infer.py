import numpy as np
from datetime import datetime

# local
import pynance

""" 
1) Read params from config files
2) get models path from params
3) Load forecasting model and regression models
4) Make prediction
5) Call strategy
6) Save results
"""

def main(path_to_cfg):
    print('\n#-------------------------- FORECASTING PROCESS --------------------#')
    # General parameters
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='infer')
    results_dir = pynance.utils.setup.get_results_dir(parameters['general']['name'])
    pynance.utils.saving.save_configobj(parameters, results_dir, 'parameters')

    (pipeliner_pred, pred_parameters), (pipeliner_regr, regr_parameters) = fetch_inference_model(parameters)
    data_dicts, index_name = check_coherence_and_return_data(pred_parameters['data'], regr_parameters['data'])

    index_dict = {index_name: data_dicts[index_name]}

    # Prediction of the next days - first on the index (market)
    # then using the regression    
    # pred_dict include the predictions on past values too (that is intermediary returns on the RNN model in this case ...)
    print('Making predictions... ', end='')
    pred_window = parameters['inference']['window']
    print('Forecasting... ', end='')
    pred_dict = pipeliner_pred.predict(index_dict, {'window': pred_window})
    print('Regressions on forcast...')
    regr_dict = pipeliner_regr.predict(pred_dict, {})

    # Dealing with strategy
    print('Loading cointegration pairs...')
    df_coint = pynance.coint.load_coint_file(parameters['general']['coint_name'])

    selected_feature = parameters['strategy']['feature']
    regr_dict_arr = {}
    for key, df_ in regr_dict.items():
        regr_dict_arr[key] = df_[selected_feature].values[-pred_window:]
    
    print('Compute best actions...')
    df_best_action = pynance.strategy.basic.get_best_action(
            df_coint,
            regr_dict_arr,
            **parameters['strategy']['best_action']
    )
    print(f'Savint to {results_dir} directory.')
    df_best_action.to_csv(results_dir / 'best_action.csv')

    # Plots - prediction using the regression models are done 
    # because it gives a hint on results quality when compared to the truth 
    # that we have.
    print(f'Plotting results... Results directory: {results_dir}')
    preds_passed = pipeliner_regr.predict(index_dict, {})    
    dates = index_dict[index_name].index.to_pydatetime()
    init_date = dates[-1]
    pred_dates = pynance.utils.dates.make_dates(pred_window, init_date)

    for key, arr_pred_future in regr_dict_arr.items():
        df_true = data_dicts[key]
        df_pred_passed = preds_passed[key]
        df_pred_passed_from_pred = regr_dict[key]
    
        x_pred_passed, y_pred_passed = dates, df_pred_passed[selected_feature].values
        x_pred_future, y_pred_future = pred_dates, arr_pred_future
        x_true, y_true = dates, df_true[selected_feature].values 
        x_full_pred, y_full_pred = dates[1:], df_pred_passed_from_pred[selected_feature].values[:-pred_window]
    
        pynance.utils.plot.plot_stock_values({
                'true': (x_true, y_true),
                'future': (x_pred_future, y_pred_future),
                'regr - true past': (x_pred_passed, y_pred_passed),
                'regr - pred past': (x_full_pred, y_full_pred)
            },
            results_dir / f'{key}.png')

    pynance.utils.plot.plot_stock_values({
        'true': (dates, index_dict[index_name][selected_feature].values),
        'pred': (np.concatenate((dates[1:], pred_dates), axis=0), pred_dict[index_name][selected_feature].values)
        },
        results_dir / f'{index_name}.png'
    )

    print('------------------ END -----------------\n')

    return True

def fetch_inference_model(parameters):
    prediction_model = parameters['inference']['forecasting_model']
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
    train_parameters['data']['start_date'] = pynance.utils.dates.get_start_date(
        infer_parameters['inference']['start_prediction_date'],
        - infer_parameters['inference']['training_window']
        )
    train_parameters['data']['end_date'] = infer_parameters['inference']['start_prediction_date']
    return train_parameters


def check_coherence_and_return_data(pred_data, regr_data):
    dict_stocks_pred = pynance.utils.setup.setup_data_section(pred_data) 
    dict_stocks_regr = pynance.utils.setup.setup_data_section(regr_data)
    assert(len(dict_stocks_pred) == 1)
    index_name = list(dict_stocks_pred.keys())[0]
    assert(index_name in dict_stocks_regr.keys())
    return dict_stocks_regr, index_name

