from pynance.utils import user
import pandas_datareader.data as web 
import yfinance as yf

# Return the stock tag of companies for a category
def read_txt(name, verbose=False):
    x=[]
    data_path = user.get_path_to_data()
    with open(data_path / (name + '.txt'), 'r') as fp:
        for line in fp:
            x.append(line[:-1])
    if verbose:
        print('Done')
    return x

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


def get_financial_datas(x, start = '1999-01-01', conversion = True):
    """ Usage : get_financial_datas(x, start = '1999-01-01', conversion = True)

    Args:
        x (list): list of stock labels in string format
        start (str, optional): Start date. Defaults to '1999-01-01'.
        conversion (bool, optional): Converts from Dollars or not. Defaults to True.

    Returns:
        _type_: _description_
    """
    stk_data = {}
    for i in x:
        stk_data[i] = yf.download(i, start = start)

    if conversion:
        conversion_us = 'DEXUSEU'
        ccy_data = web.DataReader(conversion_us, 'fred', start=start)
        ccy_data = 1/ccy_data


        def f(x):
            if x.name != 'Volume':
                x = x*ccy_data['DEXUSEU']
            return x 
    
    for i in x:
        if conversion :
            stk_data[i] = stk_data[i].apply(lambda x : f(x))
        stk_data[i][['Open', 'High', 'Low', 'Close', 'Adj Close']] = stk_data[i][['Open', 'High', 'Low', 'Close', 'Adj Close']].pct_change()*100

    return stk_data