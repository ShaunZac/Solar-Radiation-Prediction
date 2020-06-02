# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# initializing dictionary that will contain each year's data
df = {}

# reading all data into df
for i in range(15):
    df[str(i)] = pd.read_csv('data/19279_18.95_72.85_20{:02d}.csv'.format(i), 
                             header = 2)

def autocorrelation(data, k):
    """
    Parameters
    ----------
    data : ndarray
        The input data from which autocorrelation needs to be calculated.
    k : int
        The offset of hours by which autocorrelation needs to be calculated.

    Returns
    -------
    r : float
        Returns the autocorrelation of the data.
    """
    var = np.var(data)
    mean = np.mean(data)
    N = len(data) # 8760 in this case
    
    r = np.sum((data[k:] - mean) * (data[:N-k] - mean)) / ((N - k)*var)
    
    return r

def normalizeSigmoid():
    
    return

def normalizeRatio():
    
    return
