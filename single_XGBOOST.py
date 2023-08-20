#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:35:46 2023

@author: air
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error


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

'''
only use lag1...lag48 of gas, date, time, sunlight, passengers, atms
'''
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


'''
use the grid search to find the hyperparameters of XGBOOST.
Choose one hyperparameter to be tuned for each time
'''
cv_params = {'n_estimators': np.linspace(20, 100, 11, dtype=int)}#20
cv_params = {'max_depth': np.linspace(5, 10, 6, dtype=int)}#5
cv_params ={'min_child_weight':list(range(1,6,1)) }#5
cv_params = {'gamma': np.linspace(0, 0.1, 10)}#0
cv_params = {'reg_lambda': np.linspace(0, 100, 11)} #70
cv_params = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100,150,200]} #100
cv_params = {'subsample': np.linspace(0.5, 1, 6)}#1
cv_params = {'colsample_bytree': np.linspace(0.5, 1, 6)}#1
cv_params = {'learning_rate': np.linspace(0.04, 0.3, 11)} 

X = train1.drop('usage_kWh',axis=1)
y =  train1[['usage_kWh']]

params = {
 'learning_rate' : 0.09,
 'n_estimators' :80,
 'max_depth' : 6,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':100,
 'reg_alpha': 80}

tscv = TimeSeriesSplit(n_splits=3)
regress_model = XGBRegressor(**params) 
gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=tscv,scoring='neg_mean_absolute_error')
gs.fit(X, y) 
print("参数的最佳取值：:", gs.best_params_)


# After geting the hyperparameters selected using grid search, then train the model
tuned_gbr = xgb.XGBRegressor(**params)
tuned_gbr.fit(train1_X, train1_y, eval_set=[(train1_X,  train1_y),(val1_X, val1_y)])#train:60 val:67

# Test model
y_pre = tuned_gbr.predict(X_test) # 67.17 
mean_absolute_error(y_test, y_pre)
mean_absolute_percentage_error(y_test, y_pre) #10.67%
np.sqrt(mean_squared_error(y_pre,y_test))


'''
only use lag1...lag48 of gas, date, time, sunlight
'''
train1.info()
train2 = train1.drop(['passengers','atms'],axis=1)
train2_X, train2_y, val2_X, val2_y = split_train_val(train2,0.8,'usage_kWh')

test2 = test1.drop(['passengers','atms'],axis=1)
X_test = test2.dropna().drop(['usage_kWh'],axis=1)
y_test = test2.dropna()['usage_kWh']


params = {
 'learning_rate' : 0.09,
 'n_estimators' :80,
 'max_depth' : 6,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':100,
 'reg_alpha': 80}


tuned_gbr = xgb.XGBRegressor(**params)
tuned_gbr.fit(train2_X, train2_y, eval_set=[(train2_X,  train2_y),(val2_X, val2_y)])#train:60 val:67

y_pre = tuned_gbr.predict(X_test) # 67.13
mean_absolute_error(y_test, y_pre) # MAE
mean_absolute_percentage_error(y_test, y_pre) #10.61% MAPE
np.sqrt(mean_squared_error(y_pre,y_test)) #RMSE


'''
only use lag1...lag48 of gas, date, time, sunlight, atms
'''
train1.info()
train3 = train1.drop(['passengers'],axis=1)
train3_X, train3_y, val3_X, val3_y = split_train_val(train3,0.8,'usage_kWh')

test3 = test1.drop(['passengers'],axis=1)
X_test = test3.dropna().drop(['usage_kWh'],axis=1)
y_test = test3.dropna()['usage_kWh']


params = {
 'learning_rate' : 0.09,
 'n_estimators' :80,
 'max_depth' : 6,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':100,
 'reg_alpha': 80}


tuned_gbr = xgb.XGBRegressor(**params)
tuned_gbr.fit(train3_X, train3_y, eval_set=[(train3_X,  train3_y),(val3_X, val3_y)])#train:59 val:67

y_pre = tuned_gbr.predict(X_test) # 66.92
mean_absolute_error(y_test, y_pre) # MAE
mean_absolute_percentage_error(y_test, y_pre) #10.60% MAPE
np.sqrt(mean_squared_error(y_pre,y_test)) #RMSE


'''
only use lag1...lag48 of gas, date, time, sunlight, passengers
'''
train1.info()
train4 = train1.drop(['atms'],axis=1)
train4_X, train4_y, val4_X, val4_y = split_train_val(train4,0.8,'usage_kWh')

test4 = test1.drop(['atms'],axis=1)
X_test = test4.dropna().drop(['usage_kWh'],axis=1)
y_test = test4.dropna()['usage_kWh']


params = {
 'learning_rate' : 0.09,
 'n_estimators' :80,
 'max_depth' : 6,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':100,
 'reg_alpha': 80}


tuned_gbr = xgb.XGBRegressor(**params)
tuned_gbr.fit(train4_X, train4_y, eval_set=[(train4_X,  train4_y),(val4_X, val4_y)])#train:62 val:68

y_pre = tuned_gbr.predict(X_test) # 68.11
mean_absolute_error(y_test, y_pre)
mean_absolute_percentage_error(y_test, y_pre)
np.sqrt(mean_squared_error(y_pre,y_test))


'''
only use lag1...lag48 of gas, date, time
'''
train1.info()
train5 = train1.drop(['passengers','atms','sunlight_duration_minutes'],axis=1)
train5_X, train5_y, val5_X, val5_y = split_train_val(train5,0.8,'usage_kWh')

test5 = test1.drop(['passengers','atms','sunlight_duration_minutes'],axis=1)
X_test = test5.dropna().drop(['usage_kWh'],axis=1)
y_test = test5.dropna()['usage_kWh']


params = {
 'learning_rate' : 0.09,
 'n_estimators' :80,
 'max_depth' : 6,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':100,
 'reg_alpha': 80}


tuned_gbr = xgb.XGBRegressor(**params)
tuned_gbr.fit(train5_X, train5_y, eval_set=[(train5_X,  train5_y),(val5_X, val5_y)])#train:62 val:68

y_pre = tuned_gbr.predict(X_test) # 67.190787
mean_absolute_error(y_test, y_pre)
mean_absolute_percentage_error(y_test, y_pre)#0.10781
np.sqrt(mean_squared_error(y_pre,y_test))


'''
only use lag1...lag48 of gas
'''
train1.info()
train6 = train1.drop(['passengers','atms','sunlight_duration_minutes','usage_dateID_zulu','usage_timeID_zulu'],axis=1)
train6_X, train6_y, val6_X, val6_y = split_train_val(train6,0.8,'usage_kWh')

test6 = test1.drop(['passengers','atms','sunlight_duration_minutes','usage_dateID_zulu','usage_timeID_zulu'],axis=1)
X_test = test6.dropna().drop(['usage_kWh'],axis=1)
y_test = test6.dropna()['usage_kWh']


params = {
 'learning_rate' : 0.09,
 'n_estimators' :80,
 'max_depth' : 6,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':100,
 'reg_alpha': 80}


tuned_gbr = xgb.XGBRegressor(**params)
tuned_gbr.fit(train6_X, train6_y, eval_set=[(train6_X,  train6_y),(val6_X, val6_y)])#train:62 val:68

y_pre = tuned_gbr.predict(X_test) # 68.224
mean_absolute_error(y_test, y_pre)
mean_absolute_percentage_error(y_test, y_pre)#0.111474
np.sqrt(mean_squared_error(y_pre,y_test))





