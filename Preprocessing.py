# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# initializing dictionary that will contain each year's data
df = {}

# reading all data into df
for i in range(15):
    df[str(i)] = pd.read_csv('data/19279_18.95_72.85_20{:02d}.csv'.format(i), 
                             header = 2)

def autocorrelation(df):
   
    return

def normalizeSigmoid():
    
    return

def normalizeRatio():
    
    return
