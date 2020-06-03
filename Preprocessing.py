# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    
    mu = np.mean(df,axis=0)
    SD = np.std(df, axis = 0)
    
    return 1/(1 + np.exp(-(df-mu)/SD))

def initializeInputOutput(df, predict_hour, save = True):
    
    # making the full dataframe
    full_df = df['0'].copy()
    for i in range(1, 15):
        full_df = full_df.append(df[str(i)].copy())
    full_df = full_df.reset_index(drop = True)
    
    # dropping unnecessary columns
    full_df = full_df.drop(['DNI', 'GHI', 'Clearsky DHI', 'Clearsky DNI', 'Minute',
                            'Clearsky GHI', 'Snow Depth', 'Fill Flag'], axis = 1)
    # dropping the first few rows since prediction requires
    # 'predict_hour' number of prev data
    y = full_df['DHI'].iloc[predict_hour:].copy()
    X = full_df.copy()
    X['DHI'] = X['DHI'].shift(predict_hour)
    X = X.dropna()
    
    # split into train test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/15,
                                                        random_state = 0)
    
    # drop all training examples before 5 am and after 8 pm
    y_train = y_train.drop(index = X_train[(X_train['Hour'] >= 20) | (X_train['Hour'] <= 5)].index)
    X_train = X_train[(X_train['Hour'] < 20) & (X_train['Hour'] > 5)]
    
    # normalizing the train and test inputs
    X_train = normalizeSigmoid(X_train)
    X_test = normalizeSigmoid(X_test)
    
    if save:
        X_train.to_csv('IO/X_train.csv')
        y_train.to_csv('IO/y_train.csv')
        X_test.to_csv('IO/X_test.csv')
        y_test.to_csv('IO/y_test.csv')
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = initializeInputOutput(df, 1)
