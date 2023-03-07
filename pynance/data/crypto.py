import ccxt
import pandas as pd
import pynance

# --------------- Stock Time Series --------------- #
def get_exchange_object_from_name(name):
    switch={
        "binance":ccxt.binance,
        "bittrex":ccxt.bittrex,
        "bitforex":ccxt.bitforex,
        "okcoin":ccxt.okcoin,
        "bitfinex":ccxt.bitfinex,
        "kraken":ccxt.kraken,
        "kucoin":ccxt.kucoin,
        "bybit":ccxt.bybit,
        "okx":ccxt.okx,
        "coinbase":ccxt.coinbase
    }
    return switch.get(name, "Invalid input")() 
    # parenthesis to instantiate the class and return the object
    # we don't want to instantiate class before

def get_crypto_data(exchange, paire, timeframe, limit):
    exchange = get_exchange_object_from_name(exchange)
    ohlcv = exchange.fetch_ohlcv(paire, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['TIMESTAMP','OPEN','HIGH', 'LOW', 'CLOSE',' VOLUME'])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='ms')
    return format_crypto_df(df)

def format_crypto_df(df):
    df.rename(
        columns = {
            'TIMESTAMP': 'date',
            'OPEN': 'Open', 
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': pynance.utils.conventions.close_name,
            'VOLUME': 'Volume'
        }, inplace=True)
    df.set_index('date', inplace=True, drop=True)
    return df

# --------------- Real time --------------------- #
def get_bid_ask(exchange, paire, limit=20, verbose=False):
    i=0
    time_list = []
    bid_list = []
    ask_list = []
    exchange = get_exchange_object_from_name(exchange)
    while i < limit:
        time, bid, ask = get_last_bid_ask(exchange, paire)
        time_list.append(time)
        bid_list.append(bid)
        ask_list.append(ask)
        if verbose:
            print('Time =',time[i],'Bid =',bid[i],'Ask =',ask[i])
        i+=1

def get_last_bid_ask(exchange, paire):
    exchange_book = exchange.fetch_order_book((paire))
    last_bid = exchange_book["bids"][0][0]
    last_ask = exchange_book["asks"][0][0]
    now = exchange.milliseconds()
    return exchange.iso8601(now), last_bid, last_ask