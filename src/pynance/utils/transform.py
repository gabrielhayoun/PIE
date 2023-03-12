import numpy as np

def get_sliding_windows(data, window):
    x = []
    y = []

    for i in range(len(data)-window-1):
        _x = data[i:(i+window)]
        _y = data[i+window]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)
