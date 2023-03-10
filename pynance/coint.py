# local
import pynance
import pandas as pd
import datetime

def main(path_to_cfg):
    # General parameters
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='coint')
    
    feature = parameters['data']['feature']
    data_dicts = pynance.utils.setup.setup_data_section(parameters['data'])

    df = convert_dict_to_df_with_feature(data_dicts, feature)
    significance=parameters['data']['test_level']
    df_previous_coint = load_coint_file(parameters['saving_name'])
    length = len(df_previous_coint)
    score_matrix, pvalue_matrix = pynance.strategy.cointegration.test_cointegration(df.values)

    format = '%Y-%m-%d'
    start_date = parameters['data']['start_date']
    end_date = parameters['data']['end_date']
    now = datetime.datetime.now().strftime(format=format)
    tickers = df.columns # always sorted
    rows = []
    for i, ticker1 in enumerate(tickers):
        for j in range(i+1, len(tickers)):
            ticker2 = tickers[j]
            score = score_matrix[i, j]
            pvalue = pvalue_matrix[i, j]
            is_coint = False
            if(pvalue < significance):
                is_coint = True
            
            new_row = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'p-value': pvalue,
                'score': score,
                'test-level': significance,
                'is cointegrated': is_coint,
                'start date': start_date,
                'end date': end_date,
                'test date': now
            }
            rows.append(new_row)
    new_df = pd.DataFrame.from_records(rows)
    df_previous_coint = pd.concat([df_previous_coint, new_df], ).drop_duplicates(
        subset=['ticker1', 'ticker2'], keep='last').sort_values(
        by=['ticker1', 'ticker2']).reset_index(inplace=False, drop=True)

    save_coint(df_previous_coint, parameters['saving_name'])

    nb_new_pairs = len(df_previous_coint)-length
    print('Added {} new pairs.'.format(nb_new_pairs))
    print('Updated {} pairs.'.format(len(new_df) - nb_new_pairs))

def convert_dict_to_df_with_feature(data_dicts, feature):
    df = pd.DataFrame()

    for ticker, df_ in data_dicts.items():
        series = df_[feature]
        df[ticker] = series
    
    old_length = len(df)
    df.dropna(axis=0, inplace=True)
    length = len(df)
    print('Df had {} NA rows. Removing them. Remaining rows: {}.'.format(old_length-length, length))
    df.sort_index(axis=1, ascending=True, inplace=True) # sort columns in place
    return df
    
def load_coint_file(name):
    path = pynance.utils.user.get_path_to_results() / f'{name}.csv'
    if(path.is_file()):
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.DataFrame(columns=['ticker1', 'ticker2', 'p-value', 'score', 'test-level', 'is cointegrated', 'start date', 'end date', 'test date'])
    # df.set_index(['ticker1', 'ticker2'], drop=False, append=False, inplace=True)
    return df

def save_coint(df, name):
    path = pynance.utils.user.get_path_to_results() / f'{name}.csv'
    print(f'Saving path: {path}.')
    df.to_csv(path)