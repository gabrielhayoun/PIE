import ccxt
import pandas as pd
import pynance
import numpy as np
import time
import tqdm

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

def get_crypto_data(exchange, ticker, timeframe, limit):
    exchange = get_exchange_object_from_name(exchange)
    ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
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
def get_bid_ask(exchange, crypto1, crypto2, limit=20):
    bid1_list = []
    bid2_list = []
    exchange = get_exchange_object_from_name(exchange)
    for i in tqdm.tqdm(range(limit)):
        t1 = time.time()
        _, bid1, _ = get_last_bid_ask(exchange, crypto1)
        _, bid2, _ = get_last_bid_ask(exchange, crypto2)
        bid1_list.append(bid1)
        bid2_list.append(bid2)
        t2 = time.time()
        sleep_time = 1 - (t2 - t1)
        if(sleep_time > 0):
            time.sleep(sleep_time)
    return np.array(bid1_list), np.array(bid2_list)

def get_last_bid_ask(exchange, paire):
    exchange_book = exchange.fetch_order_book(paire, limit=1)
    last_bid = exchange_book["bids"][0][0]
    last_ask = exchange_book["asks"][0][0]
    now = exchange.milliseconds()
    return exchange.iso8601(now), last_bid, last_ask