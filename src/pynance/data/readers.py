from pynance.utils import user
from pynance.utils import conventions
import pandas_datareader.data as web 
import yfinance as yf

# Return the stock tag of companies for a category
def read_txt(name):
    lines = []
    data_path = user.get_path_to_data()
    with open(data_path / (name + '.txt'), 'r') as fp:
        for line in fp:
            lines.append(line.rstrip())
    return lines

# Write a stock name in the appropriate list
# tag_list => ['IBM', 'GOOGL']
# name => tech
def write_stock_txt(tag_list, name, verbose = False):
    try:
        x = read_txt(name)
    except:
        x = []

    data_path = user.get_path_to_data()
    with open(data_path / (name + '.txt'), 'a') as fp:
        for item in tag_list:
            if item in x:
                print(f"{item} is already in the category {name} and won't be added again")
            else:
                try:
                    web.DataReader(item, 'yahoo')
                except:
                    print(f"{item} does not seem to be a stock action")
                    continue
                # write each item on a new line
                fp.write("%s\n" % item)
    if verbose:
        print('Done')

# Write an idx name in the appropriate list
# tag_list => ['SP500', 'DJIA', 'VIXCLS']
# name => idx or ccy
def write_idx_txt(tag_list, name, verbose = False):
    try:
        x = read_txt(name)
    except:
        x = []

    data_path = user.get_path_to_data()
    with open(data_path / (name + '.txt'), 'a') as fp:
        for item in tag_list:
            if item in x:
                print(f"{item} is already in the category {name} and won't be added again")
            else:
                try:
                    web.DataReader(item, 'fred')
                except:
                    print(f"{item} does not seem to be a stock action")
                    continue
                # write each item on a new line
                fp.write("%s\n" % item)
    if verbose:
        print('Done')

def get_financial_datas(x, start = '1999-01-01', end=None,
                        conversion = True, remove_nan=True,
                        return_type='returns'):
    """ Usage : get_financial_datas(x, start = '1999-01-01', conversion = True)

    Args:
        x (list): list of stock labels in string format
        start (str, optional): Start date. Defaults to '1999-01-01'.
        conversion (bool, optional): Converts from Dollars or not. Defaults to True.

    Returns:
        dict: Dictionnary containing for each stock entry a dataframe with 'Open', 'High',
             'Low', 'Close', 'Adj Close' columns and the dates as index.
    """
    assert(return_type in ["returns", "raw"])
    stk_data = {}
    for i in x:
        stk_data[i] = yf.download(i, start=start, end=end, progress=False)
        print(f'Downloaded {len(stk_data[i])} data for {i} from {start} to {end}.')
    if conversion:
        conversion_us = 'DEXUSEU'
        ccy_data = web.DataReader(conversion_us, 'fred', start=start, end=end)
        ccy_data = 1/ccy_data


        def f(x):
            if x.name != 'Volume':
                x = x*ccy_data['DEXUSEU']
            return x 
    
    for i in x:
        if conversion :
            stk_data[i] = stk_data[i].apply(lambda x : f(x))
        if(return_type == 'returns'):
            stk_data[i][['Open', 'High', 'Low', 
                        conventions.close_name,
                        'Adj Close']] = stk_data[i][
                            ['Open', 'High', 'Low',
                            'Close', 'Adj Close']].pct_change() #*100
            if(remove_nan): # after pct_change because pct_change add NaN for first row
                stk_data[i].dropna(inplace=True, axis=0)
        elif(return_type == 'raw'):
            stk_data[i][['Open', 'High', 'Low', 
                        conventions.close_name,
                        'Adj Close']] = stk_data[i][
                            ['Open', 'High', 'Low',
                            'Close', 'Adj Close']] 
            stk_data[i].interpolate(method='linear', inplace=True, axis=0)
        # stk_data[i][conventions.date_name] = stk_data[i].index
        if(remove_nan): # after pct_change because pct_change add NaN for first row
            stk_data[i].dropna(inplace=True, axis=0)
    return stk_data