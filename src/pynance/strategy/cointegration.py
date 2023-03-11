import numpy as np
from statsmodels.tsa.stattools import coint
import logging

def test_cointegration(data):
    if(data.shape[0] < data.shape[1]):
        logging.warning(f'Data shape is: {data.shape}.'
                          'Expected: number of samples x number of tickers.'
                          'Such as returned by df.values')
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    # pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[:, i]
            S2 = data[:, j]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            # if pvalue < significance:
            #     pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix #, pairs