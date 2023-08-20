#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:39:00 2023

@author: air
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import xgboost as xgb
import random
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, col_names=None):
    

    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        col_names: List of column names for the data.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    #n_vars = data.shape[1]
    df = pd.DataFrame(data, columns=col_names)
    cols, names = [], []
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{col}(t-{i})') for col in col_names]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'{col}(t)') for col in col_names]
        else:
            names += [(f'{col}(t+{i})') for col in col_names]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg

data = pd.read_csv('/Users/air/Desktop/imputed_total_data.csv')
data.drop('Unnamed: 0',axis=1, inplace=True)
data5 = data.copy()
data5 = data5[['sunlight_duration_minutes','usage_kWh','passengers','atms']] 
data5.columns
values = data5.values

data5 = series_to_supervised(values,n_in=48,col_names=data5.columns)

# split test set 
X, X_test, y, y_test = train_test_split(data5.drop('usage_kWh(t)',axis = 1),data5[['usage_kWh(t)']], test_size=0.1, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)


params = {
 'learning_rate' : 0.1,
 'n_estimators' :95,
 'max_depth' : 6,
 'min_child_weight' : 1,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 0.95,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : 'mae',
 'reg_lambda':80,
 'reg_alpha':150}

tuned_gbr = xgb.XGBRegressor(**params)
tuned_gbr.fit(X_train,y_train, eval_set=[(X_train,y_train),(X_val, y_val)],verbose=False)



# Parameters to be adjusted
tune_dic = OrderedDict()
 
tune_dic['n_estimators'] = [85,90,95,100,105]
tune_dic['max_depth']= [4,5,6,7]
tune_dic['min_child_weight'] = [1,2,3]
tune_dic['subsample']=[0.85,0.9,0.95,1]
tune_dic['colsample_bytree']= [0.9,0.95,1]
tune_dic['learning_rate']= [0.07,0.1,0.13, 0.15]
tune_dic['gamma']= [0.00,0.05,0.10]
tune_dic['reg_lambda'] = [75,80,85]
tune_dic['reg_alpha'] = [120,150,180]

lengths = [len(lst) for lst in tune_dic.values()] # num of parameters been tunned
combs=1
for i in range(len(lengths)):
    combs *= lengths[i]
print('Number of Combinations: {:16d}'.format(combs))  
maxiter=200
columns=[*tune_dic.keys()]+['MAE','Best MAE']
results = pd.DataFrame(index=range(maxiter), columns=columns) 

def do_train(cur_choice, params, train_X,train_Y,valid_X,valid_y):
    '''
    train the model with selected parameters 
    
    '''
    print('parameters:')
    for (key,value) in cur_choice.items():
        print(key,': ',value,' ',end='')
        params[key]=value
    print('\n')    
    
    evallist  = [(X_train,y_train), (X_val,y_val)]
    model = xgb.XGBRegressor(**params)
    model.fit(train_X,train_Y, eval_set=evallist,verbose=False)  
    mae = model.evals_result()['validation_1']['mae'][-1]
    return mae,model

def next_choice(cur_params=None):

  if cur_params:
        li = ['n_estimators','max_depth','min_child_weight','subsample','colsample_bytree','learning_rate','gamma','reg_lambda','reg_alpha']
        filtered_params = {key: value for key, value in cur_params.items() if key in li}
        choose_param_name, cur_value = random.choice(list(filtered_params.items()))       
        all_values =  list(tune_dic[choose_param_name]) # 所选参数的所有值
        cur_index = all_values.index(cur_value) # 选定参数的当前索引
        
        if cur_index==0: 
            # If it is the first in the range, select the second
            next_index=1
        elif cur_index==len(all_values)-1: 
            # If it is the last one in the range, the previous one is selected
            next_index=len(all_values)-2
        else: 
            # Otherwise randomly select left or Rvalue
            direction=np.random.choice([-1,1])
            next_index=cur_index + direction
 
        next_params = dict((k,v) for k,v in cur_params.items())
        next_params[choose_param_name] = all_values[next_index]
        
        # update the value of the selected parameter
        print('Parameter selection moves to : {:10s}: from {:6.2f} to {:6.2f}'.
              format(choose_param_name, cur_value, all_values[next_index] ))
  else: 
       # Generate random combinations of parameters
        next_params=dict()
        next_params['booster'] = 'gbtree'
        next_params['objective'] = 'reg:squarederror'
        next_params['eval_metric'] = 'mae'
        for i in range(len(tune_dic)):
            key = [*tune_dic.keys()][i] 
            values = [*tune_dic.values()][i]
            next_params[key] = np.random.choice(values)
  return(next_params)



#Simulated Annealing
import time
t0 = time.process_time()
T=100
best_params = dict() # The dictionary is initialized to hold the best parameters
best_mae = tuned_gbr.evals_result()['validation_1']['mae'][-1]  # initialize optimal F-score
prev_mae = tuned_gbr.evals_result()['validation_1']['mae'][-1]   # initialize the previou F-score
prev_choice = params   # initialize previous selected parameters
weights = list(map(lambda x: 10**x, [0,1,2,3,4])) # The weight of the hash function
hash_values=set()
 

for iter in range(maxiter):
    T = 100*0.99**iter
    print('\nIteration = {:5d}  T = {:12.6f}'.format(iter,T))
    # Find the next selection of a parameter that has not been accessed before
    while True:
        # The first neighbor to select or select prev_choice
        cur_choice = next_choice(prev_choice)          
        # A selection index arranged alphabetically by parameter 
        indices=[tune_dic[name].index(cur_choice[name]) for name in sorted([*tune_dic.keys()])]      
        # Check whether the selection has already been accessed
        hash_val = sum([i*j for (i, j) in zip(weights, indices)])
        if hash_val in hash_values:
            print('\n再次访问组合 -- 再次搜索')
        else:
            hash_values.add(hash_val)
            break 
     
    # The model is trained and mae is obtained on the validation dataset
    mae,model = do_train(cur_choice, params, X_train,y_train,X_val,y_val)
    
    # save the parameters
    results.loc[iter,[*cur_choice.keys()]]=list(cur_choice.values())   
    print('MAE: {:6.2f}  previous: {:6.2f} optimal: {:6.2f}'.format(mae, prev_mae, best_mae))
 
    if mae < prev_mae:
        print('Local improvement')   
        # Accept this combination as a new starting point
        prev_mae = mae
        prev_choice = cur_choice     
        # If the F-score is good globally, the best parameter is updated
        if mae < best_mae:         
            best_mae = mae
            print('Global improvement - Best mae has been updated')
            for (key,value) in prev_choice.items():
                best_params[key]=value
                
    else: # if mae is larger than previous value   
        rnd = random.random()
        diff = mae-prev_mae
        thres = min(1,np.exp(-diff/T)) # acceptance probability
        if rnd <= thres:
            print('Worse results. MAE change: {:8.4f}  threshold: {:6.4f}  random number: {:6.4f} -> accept'.
                  format(diff, thres, rnd))
            prev_mae = mae
            prev_choice = cur_choice
        else:
            # Do not update the previous F-score and previously selected parameters
            print('Worse results. MAE change: {:8.4f}  threshold: {:6.4f}  random number: {:6.4f} -> reject'.
                 format(diff, thres, rnd))
            
    # save the results
    results.loc[iter,'MAE']=mae
    results.loc[iter,'最佳MAE']=best_mae     
print('Spend \n{:6.1f} \n'.format((time.process_time() - t0)/60))    
print('The optimal parameters are:\n')
print(best_params)

'''
use the parameters found by SA to train xgboost to predict the usage of gas.
'''
params = {
 'learning_rate' : 0.08,
 'n_estimators' :70,
 'max_depth' : 5,
 'min_child_weight' : 3,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda': 400,
 'reg_alpha': 500}

xgb.XGBRegressor(**best_params)
tuned_gbr.fit(X_train,y_train, eval_set=[(X_train,y_train),(X_val, y_val)])# train:61.19, val:64.96

y_pre = tuned_gbr.predict(X_test)
mean_absolute_error(y_test,y_pre)#64.34


