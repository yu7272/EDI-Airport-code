#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 21:42:39 2023

@author: air
"""


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('imputed_total_data.csv')
data.drop('Unnamed: 0',axis=1, inplace=True)
data.info()


'''
Divide the training and test set, select data in 2023 as test set
 
'''
data['datetime'] = pd.to_datetime(data['usage_dateID_zulu'].astype(str), format='%Y%m%d')
data = data.set_index('datetime')
train = data.loc[data.index < '2023-01-01']
test = data.loc[data.index >= '2023-01-01']
train.info()


'''
split training and validation set
'''
def split_train_val(data,num,variable_y):
    
    train_num = int(len(data)*num)
    train_X = data.drop(f'{variable_y}',axis=1).iloc[:train_num,:]
    train_y = data[[f'{variable_y}']].iloc[:train_num,:]
    
    val_X = data.drop(f'{variable_y}',axis=1).iloc[train_num:,:]
    val_y = data[[f'{variable_y}']].iloc[train_num:,:]
    
    return train_X, train_y, val_X, val_y



train_X = train.drop('usage_kWh',axis=1).loc[train.index <= '2022-03-29']
train_y = train[['usage_kWh']].loc[train.index <= '2022-03-29']

val_X = train.drop('usage_kWh',axis=1).loc[train.index > '2022-03-29']
val_y = train[['usage_kWh']].loc[train.index > '2022-03-29']


test_X = test.drop('usage_kWh',axis=1)
test_y = test[['usage_kWh']]

test_X.drop(['passengers','usage_dateID_zulu','usage_timeID_zulu'],axis=1,inplace=True)


train_X.info()

train_X.drop(['passengers','usage_dateID_zulu','usage_timeID_zulu'],axis=1,inplace=True)
val_X.drop(['passengers','usage_dateID_zulu','usage_timeID_zulu'],axis=1,inplace=True)


'''
reset the index of data
'''

# Create a new datetime index with 30-minute frequency for the entire data range
train_date_range = pd.date_range(start=data.index.min(), end='2022-03-29 23:30:00', freq='30T')
val_date_range = pd.date_range(start='2022-03-30 00:00:00', end='2022-12-31 23:30:00', freq='30T')
# Set this new index to train_X, train_y, val_X, and val_y

train_X.reset_index(inplace = True)
train_y.reset_index(inplace = True)
val_X.reset_index(inplace = True)
val_y.reset_index(inplace = True)

train_X.drop('datetime',axis=1,inplace=True)
train_y.drop('datetime',axis=1,inplace=True)
val_X.drop('datetime',axis=1,inplace=True)
val_y.drop('datetime',axis=1,inplace=True)


train_X = train_X.set_index(train_date_range)
train_y = train_y.set_index(train_date_range)
val_X = val_X.set_index(val_date_range)
val_y = val_y.set_index(val_date_range)

test_date_range = pd.date_range(start='2023-01-01 00:00:00', end='2023-06-13 23:00:00', freq='30T')
test_X.reset_index(inplace = True)
test_y.reset_index(inplace = True)
test_X.drop('datetime',axis=1,inplace=True)
test_y.drop('datetime',axis=1,inplace=True)
test_X = test_X.set_index(test_date_range)
test_y = test_y.set_index(test_date_range)


'''
ADF-test to check stationary : d=1,0
'''
# trainy use original
res = sm.tsa.adfuller(train_y,regression='ct')
print('p-value:{}'.format(res[1]))

# valy need to use difference
res = sm.tsa.adfuller(val_y,regression='ct')
print('p-value:{}'.format(res[1]))

res = sm.tsa.adfuller(val_y.diff().dropna(),regression='c')
print('p-value:{}'.format(res[1]))


'''
acf and pacf : p=1,2

'''
#we use tra.diff()(differenced data), because this time series is unit root process.
fig,ax = plt.subplots(2,1,figsize=(20,10))
fig = sm.graphics.tsa.plot_acf(train_y.diff().dropna(), lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(train_y.diff().dropna(), lags=50, ax=ax[1])
plt.show()



# construct ARIMAX model
sarimax = sm.tsa.statespace.SARIMAX(train_y,order=(48,1,1),seasonal_order=(0,0,0,0),exog = train_X,
                                enforce_stationarity=False, enforce_invertibility=False,freq='30T').fit()
sarimax.summary()

# Check whether the residuals have autocorrelation
res = sarimax.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()

# Check validation MAE 
y_pred = sarimax.predict(start=val_y.index[0], end=val_y.index[-1],exog = val_X)
mean_absolute_error(val_y,y_pred)


# Check train MAE
train_pred = sarimax.predict(start=train_y.index[0], end=train_y.index[-1])
mean_absolute_error(train_y,train_pred)

# Check Test MAE
test_pred = sarimax.forecast(steps=1, exog=test_X.iloc[20,])
mean_absolute_error(test_y.iloc[20],test_pred)

