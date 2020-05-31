# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv('19279_18.95_72.85_2000.csv', header = 2)
for i in range(1, 15):
    temp_df = pd.read_csv('19279_18.95_72.85_20{:02d}.csv'.format(i), header = 2)
    df['DHI'] += temp_df['DHI']
df['DHI'] /= 15
df

#See the change in radiation with seasons, implemented in the paper
plt.figure()
plt.plot(df['DHI'])
plt.xlabel('Hour')
plt.ylabel("Solar Radiation ($w/m^2$)")


plt.figure()
dates = pd.to_datetime(df.Year*10000+df.Month*100+df.Day,format='%Y%m%d')
df['Day of Year'] = dates.dt.dayofyear
cols = [str(i) for i in range(1, 24)]
data = pd.DataFrame(columns = cols)
for group, val in df.groupby('Hour'):
    data[group] = val['DHI'].reset_index(drop=True)
#data = data.dropna(axis=0)
data = data.drop([ '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'], axis=1)
data.head()
sns.heatmap(data.to_numpy(), cmap="CMRmap")
