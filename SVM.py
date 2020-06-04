# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from Preprocessing import initializeInputOutput

# initializing dictionary that will contain each year's data
df = {}

# reading all data into df
for i in range(15):
    df[str(i)] = pd.read_csv('data/19279_18.95_72.85_20{:02d}.csv'.format(i), 
                             header = 2)
mae = []
mape = []
for i in range(2, 16):
    # getting the train test data
    X_train, X_test, y_train, y_test = initializeInputOutput(df, i)
    
    model = SVR()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    
    y_pred = pd.Series(y_pred)
    y_test = pd.Series(y_test).reset_index(drop=True)
    
    print(str(i) + ".", "MAE:", mean_absolute_error(y_test, y_pred))
    mae.append(mean_absolute_error(y_test, y_pred))
    
    y_pred_mape = y_pred.drop(index = y_test[y_test == 0].index)
    y_test_mape = y_test.drop(index = y_test[y_test == 0].index)
    
    mape_now = np.mean(abs(y_test_mape - y_pred_mape)/y_pred_mape)
    
    print(str(i) + ".", "MAPE:", mape_now)
    mape.append(mape_now)

def plotCorrelation():
    # plotting the correlation between actual and predicted DHI
    plt.figure()
    plt.scatter(y_pred, y_test, marker = '.', color = 'g', linewidths = 0.01, label = 'Actual value')
    plt.plot(y_pred, y_pred, color = 'r', label = 'Fitted line')
    plt.xlabel("Predicted radiation $(W/m^2)$")
    plt.ylabel("Actual radiation $(W/m^2)$")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_MAE_MAPE_years():
    x = [i for i in range(2, 16)]
    mape_percent = [i*100 for i in mape]
    
    plt.figure()
    fig, ax1 = plt.subplots()
    plt.grid(True, linestyle='-')
    
    color = 'tab:red'
    ax1.set_xlabel('Training length (years)')
    ax1.set_ylabel('MAE $(W/m^2)$', color=color)
    ax1.plot(x, mae, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('MAPE (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, mape_percent, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
def predHours(horizon):
    """
    Parameters
    ----------
    horizon : int
        The prediction horizon in hours.

    Returns
    -------
    mae : float
        The mean absolute error calculated.
    mape : float
        The mean absolute percentage error calculated.
    """
    X_train, X_test, y_train, y_test = initializeInputOutput(df, 15, horizon)
    
    model = SVR()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    
    y_pred = pd.Series(y_pred)
    y_test = pd.Series(y_test).reset_index(drop=True)
    
    print(str(i) + ".", "MAE:", mean_absolute_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    y_pred_mape = y_pred.drop(index = y_test[y_test == 0].index)
    y_test_mape = y_test.drop(index = y_test[y_test == 0].index)
    
    mape = np.mean(abs(y_test_mape - y_pred_mape)/y_pred_mape)
    
    print(str(i) + ".", "MAPE:", mape_now)
    
    return mae, mape
