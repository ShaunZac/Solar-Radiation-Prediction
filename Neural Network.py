# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import l2
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from Preprocessing import initializeInputOutput

# initializing dictionary that will contain each year's data
df = {}

# reading all data into df
for i in range(15):
    df[str(i)] = pd.read_csv('data/19279_18.95_72.85_20{:02d}.csv'.format(i), 
                             header = 2)

def myModel(input_shape):
    
    X_input = Input(input_shape)
    X = X_input
    
    X = Dense(20, activation='relu', name='fc1', kernel_regularizer=l2(0.001))(X)
    # X = Dense(10, activation='relu', name='fc2', kernel_regularizer=l2(0.001))(X)
    X = Dense(1, activation='relu', name='fc3', kernel_regularizer=l2(0.001))(X)
    
    model = Model(inputs = X_input, outputs = X, name='myModel')
    
    return model

def plotLoss(history):
    plt.figure()
    plt.plot(history.history['mean_absolute_error'], label='train')
    plt.plot(history.history['val_mean_absolute_error'], label='test')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("Mean Absolute Error $(W/m^2)$")
    plt.title("Variation of loss with epochs")
    plt.show()
    
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

mae = []
for i in range(1, 4):
    X_train, X_test, y_train, y_test = initializeInputOutput(df, 15, i)
    model = myModel(X_train.shape[1:])
    
    model.compile(loss=MeanSquaredError(), optimizer='RMSprop',
                  metrics=[MeanAbsoluteError()])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=512, verbose=0,
                        validation_data=(X_test, y_test)) 
    
    mae.append(model.evaluate(X_test, y_test, verbose = 0)[1])
    print(str(i) + ".", "MAE:", mae[i-1])
    
y_pred = model.predict(X_test)
plotCorrelation()
