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

def getCorrelationTable(df):
    """
    Parameters
    ----------
    df : dict
        Dictionary containing DataFrames of each year.

    Returns
    -------
    corr_table : pandas DataFrame
        Contains the autocorrelation table required.
    """
    # getting avg DHI of all years (because that's what is done in the paper)
    DHI = np.zeros(8760) #8760 is the number of rows
    # taking sum of all DHI
    for i in range(15):
        DHI += df[str(i)]['DHI'].to_numpy()
    # converting sum to average
    DHI /= 15
    
    # filling the correlation table
    table = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            table[i, j] = autocorrelation(DHI, 24*(3-j) + (3-i))
            
    indices = ['i - 3', 'i - 2', 'i - 1', 'i']
    cols = ['j - 3', 'j - 2', 'j - 1', 'j']
    corr_table = pd.DataFrame(table, columns = cols)
    corr_table['Hour/Day'] = indices
    corr_table = corr_table.set_index('Hour/Day')
    
    return corr_table

def normalizeSigmoid(df):
    rows=len(df)
    mu=np.sum(df,axis=0)/rows
    SD= np.sqrt(np.sum(np.multiply(df,df),axis=0)/rows-np.square(mu))
    return 1/(1+np.exp(-(df-mu)/SD))

def normalizeRatio(df):
    R_t = df['Clearsky DHI']
    r_t = []
    for i in range(len(df)):
        if R_t[i] == 0:
            r_t.append(1)
        else:
            r_t.append(R_t[i])
    data = df.div(r_t,axis = 0)
    data['Clearsky DHI'] = R_t
    return data
