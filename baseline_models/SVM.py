#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:24:14 2023

@author: air
"""

from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/Users/air/Desktop/imputed_total_data.csv')
data.drop('Unnamed: 0',axis=1, inplace=True)
data.info()
data.drop('passengers',axis = 1,inplace=True)


'''
Divide the training and test set, select data in 2023 as test set
 
'''
data['datetime'] = pd.to_datetime(data['usage_dateID_zulu'].astype(str), format='%Y%m%d')
data = data.set_index('datetime')
train = data.loc[data.index < '2023-01-01']
test = data.loc[data.index >= '2023-01-01']

train.reset_index(inplace = True)
test.reset_index(inplace = True)

def add_lags(data, lag):
    '''
    add the lags of usage of gas 
    '''
    data1 = data.copy()
    for i in range(1, lag+1):
        data1[f'lag{i}'] = data1['usage_kWh'].shift(i)
    data1.dropna(inplace=True)
    data1 = data1.drop('datetime',axis=1)
    return data1

train1 = add_lags(train,48)
train1.info()

test1 = add_lags(test,48)

X_test = test1.dropna().drop(['usage_kWh'],axis=1)
y_test = test1.dropna()['usage_kWh']


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


train1_X, train1_y, val1_X, val1_y = split_train_val(train1,0.8,'usage_kWh')
train1_X.info()

# Normalize data
scaler = StandardScaler()
train1_X_scaler =  scaler.fit_transform(train1_X)
val1_X_scaler = scaler.fit_transform(val1_X)
test1_X_scaler = scaler.fit_transform(X_test)   

'''
The performance of model with linear kernel is bad so do not choose this model

'''
#linear kernel function 
# lin_svr = SVR(kernel='linear')
# lin_svr.fit(train1_X_scaler,train1_y)
# lin_svr_pred=lin_svr.predict(val1_X_scaler)
# mean_absolute_error(val1_y, lin_svr_pred)

'''
The performance of model with polynomial kernel is bad so do not choose this model

'''
#polynomial kernel function
poly_svr = SVR(kernel='poly', degree=1, coef0=1,C=0.3)
poly_svr.fit(train1_X_scaler,train1_y)
poly_svr_pred=poly_svr.predict(val1_X_scaler)
mean_absolute_error(val1_y, poly_svr_pred)

mean_absolute_error(train1_y, poly_svr.predict(train1_X_scaler))
mean_absolute_error(y_test, poly_svr.predict(test1_X_scaler))



'''
Use grid search to choose the hyperparameters
'''
X = pd.concat([train1_X,val1_X],axis=0)
y =  pd.concat([train1_y,val1_y],axis=0)
tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    'degree': [2, 3, 4],
    'coef0': [0, 0.5, 1],
}


gs = GridSearchCV(poly_svr, param_grid, verbose=2, refit=True, cv=tscv,scoring='neg_mean_absolute_error')
gs.fit(X,y)
print("Optimal params:", gs.best_params_)

'''
SVR model with RBF kernel could perform better than choosing linear and polynomial kernel functions
'''
# RBF kerbel
rbf_svr = SVR(kernel='rbf',gamma=0.02,C=0.1)
rbf_svr.fit(train1_X_scaler,train1_y)
rbf_svr_pred=rbf_svr.predict(val1_X_scaler)

mean_absolute_error(val1_y, rbf_svr_pred)
mean_absolute_error(train1_y, rbf_svr.predict(train1_X_scaler))
mean_absolute_error(y_test, rbf_svr.predict(test1_X_scaler))
