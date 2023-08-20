#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:27:15 2023

@author: air
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

data = pd.read_csv('imputed_total_data.csv')
data.drop('Unnamed: 0',axis=1, inplace=True)
data.info()


# draw the usage of gas from 2019 to 2023
data['usage_kWh'].plot(style='.',
        figsize=(15, 5),
        title='gas consumpution')
plt.show()

#outliers
data['usage_kWh'].plot(kind='hist', bins=100)


'''
Divide the training and test set, select data in 2023 as test set
 
'''
data['datetime'] = pd.to_datetime(data['usage_dateID_zulu'].astype(str), format='%Y%m%d')
data = data.set_index('datetime')
train = data.loc[data.index < '2023-01-01']
test = data.loc[data.index >= '2023-01-01']

train.columns

#viewing: Train / Test Split
fig, ax = plt.subplots(figsize=(15, 5))
train['usage_kWh'].plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test['usage_kWh'].plot(ax=ax, label='Test Set')
ax.axvline('2023-01-01', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


# extract the daily gas consumption
train.reset_index(inplace=True)
daily_data = train.pivot(index='datetime', columns='usage_timeID_zulu', values='usage_kWh')


'''
using DBSCAN to clustering the dates according the daily gas consumption

'''
# Standardize the data
X = StandardScaler().fit_transform(daily_data)

# constrcut DBSCAN model
db = DBSCAN(eps=4.5, min_samples=385).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print(f"Silhouette Coefficient: {silhouette_score(X, labels):.3f}")

'''

#select hyperparameters : eps and min_samples in DBSCAN
best_score = -1
best_eps = 0
best_min_samples = 0

for eps in np.arange(1, 20, 0.5):
    for min_samples in range(5, 500, 5):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:  # Ensure more than one cluster is formed
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples

print("Best eps:", best_eps) #3.5 4.5
print("Best min_samples:", best_min_samples)#185 385
print("Best Silhouette Score:", best_score)# 0.22 0.237
'''

#draw the clustering results
unique_labels = {-1, 0}#set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        #xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        #xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("The results of DBSCAN")
plt.show()


#draw the usage of gas of each day in each cluster
daily_data['cluster'] = labels

# Plotting each cluster separately
unique_clusters = np.unique(labels)

plt.figure(figsize=(10, 6))
for cluster_label in unique_clusters:
    
    cluster_data = daily_data[daily_data['cluster'] == cluster_label]
    x = cluster_data.columns[:-1]
    # Plot 'usage_kWh' values for each day
    for j in cluster_data.iloc[:10,:-1].values:
        plt.plot(x, j)
    
    plt.title(f'Cluster {cluster_label} - usage_kWh Trend')
    plt.xlabel('Time')
    plt.ylabel('usage_kWh')
    plt.legend()
    
    Time_full = pd.date_range(start='00:29:59', end='23:59:59', freq='30T').time
    interval = 4  # Example: every 2 hours (because we have data every 30 minutes)
    indices_sparse = np.arange(0, len(x), interval)
    Time_sparse = [Time_full[i] for i in indices_sparse]
    # plt.xticks(ticks=indices_sparse, labels=Time_sparse, rotation=45)
    # plt.show()
     
    # Time = pd.date_range(start='00:29:59', end='23:59:59',freq='30T').time
   # plt.xticks(ticks=np.arange(len(x)), labels=Time, rotation=45)
    
    plt.gca().set_xticklabels(Time_sparse)
    plt.xticks(rotation=45)
    plt.show()


train.reset_index(inplace = True)


def add_lags(data, lag):
    '''
    add the lags of usage of gas 
    '''
    data1 = data.copy()
    for i in range(1, lag+1):
        data1[f'lag{i}'] = data1['usage_kWh'].shift(i)
    return data1


# reset the index of daily gas consumption
daily_data.reset_index(inplace = True)


def Extract_data(cluster_num,lag,variables):
    '''
    Extrct the observations in each cluster
    cluster_num: the cluster number
    lag : the number of lags of gas consumption
    variables: Do not to be considered !
    '''
    train1 = add_lags(train,lag)
    train1_cluster = train1.copy()
    merged_data = train1_cluster.merge(daily_data[['datetime','cluster']], how='inner', on='datetime')

    final_data = merged_data[merged_data['cluster']==cluster_num]
    X = final_data.dropna().drop(variables,axis=1)
    X.drop('index',axis =1,inplace=True)
    y = final_data.dropna()['usage_kWh']
    
    return X, y


# cluster 1
variables1 = ['datetime','usage_kWh','passengers','cluster']
variables2 = ['datetime','usage_kWh','atms','cluster']
variables3 = ['datetime','usage_kWh','atms','passengers','cluster']
variables4 = ['datetime','usage_kWh','atms','passengers','cluster','sunlight_duration_minutes']
variables5 = ['datetime','usage_kWh','atms','passengers','cluster','sunlight_duration_minutes','usage_dateID_zulu','usage_timeID_zulu']

X1,y1 = Extract_data(1,24,variables3)
X1.info()

'''
split training and validation set
'''
def split_train_val(X,y,num):
    
    train_num = int(len(X)*num)
    train_X = X.iloc[:train_num,:]
    train_y = y.iloc[:train_num]
    
    val_X = X.iloc[train_num:,:]
    val_y = y.iloc[train_num:]
    
    return train_X, train_y, val_X, val_y

train1_X, train1_y, val1_X, val1_y = split_train_val(X1,y1,0.8)
train1_X.columns


'''
use the grid search to find the hyperparameters of XGBOOST.
Choose one hyperparameter to be tuned for each time
'''

cv_params = {'n_estimators': np.linspace(50, 100, 6, dtype=int)}
cv_params = {'max_depth': np.linspace(5, 10, 6, dtype=int)}
cv_params ={'min_child_weight':list(range(1,6,2)) }
cv_params = {'gamma': np.linspace(0, 0.1, 10)}#0
cv_params = {'reg_lambda': np.linspace(0, 100, 11)} #0
cv_params = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100,150,200]} #1
cv_params = {'subsample': np.linspace(0.5, 1, 6)}#1
cv_params = {'colsample_bytree': np.linspace(0.5, 1, 6)}#1
cv_params = {'learning_rate': np.linspace(0.04, 0.3, 11)} 


params1 = {
 'learning_rate' : 0.1,
 'n_estimators' :80,
 'max_depth' : 5,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':120,
 'reg_alpha':5}

tscv = TimeSeriesSplit(n_splits=3)
regress_model = XGBRegressor(**params1) 
gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=tscv,scoring='neg_mean_absolute_error')
gs.fit(X1, y1) 
print("Optimal params:", gs.best_params_)

# After geting the hyperparameters selected using grid search, then train the model
tuned_gbr1 = xgb.XGBRegressor(**params1)
tuned_gbr1.fit(train1_X,train1_y, eval_set=[(train1_X,train1_y),(val1_X, val1_y)]) #58，63


# cluster 0
X2,y2 = Extract_data(0,24,variables3)

train2_X, train2_y, val2_X, val2_y = split_train_val(X2,y2,0.8)
train2_X.columns

'''
use the grid search to find the hyperparameters of XGBOOST.
Choose one hyperparameter to be tuned for each time
'''
cv_params = {'n_estimators': np.linspace(50, 100, 6, dtype=int)}#20
cv_params = {'max_depth': np.linspace(5, 10, 6, dtype=int)}#5
cv_params ={'min_child_weight':list(range(1,6,2)) }#5
cv_params = {'gamma': np.linspace(0, 0.1, 10)}#0
cv_params = {'reg_lambda': np.linspace(0, 100, 11)} #70
cv_params = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100,150,200]} #100
cv_params = {'subsample': np.linspace(0.5, 1, 6)}#1
cv_params = {'colsample_bytree': np.linspace(0.5, 1, 6)}#1
cv_params = {'learning_rate': np.linspace(0.04, 0.3, 11)} 


params2 = {
 'learning_rate' : 0.11,
 'n_estimators' :50,
 'max_depth' : 5,
 'min_child_weight' : 5,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':1,
 'reg_alpha':200}

tscv = TimeSeriesSplit(n_splits=3)
regress_model = XGBRegressor(**params2) 
gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=tscv,scoring='neg_mean_absolute_error')
gs.fit(X2, y2) 
print("Optimal params:", gs.best_params_)

# After geting the hyperparameters selected using grid search, then train the model
tuned_gbr2 = xgb.XGBRegressor(**params2)
tuned_gbr2.fit(train2_X,train2_y, eval_set=[(train2_X,train2_y),(val2_X, val2_y)])



# cluster -1: noise cluster
X3,y3 = Extract_data(-1,48,variables3)

train3_X, train3_y, val3_X, val3_y = split_train_val(X3,y3,0.8)
train3_X.columns


'''
use the grid search to find the hyperparameters of XGBOOST.
Choose one hyperparameter to be tuned for each time
'''
cv_params = {'n_estimators': np.linspace(50, 100, 6, dtype=int)}#20
cv_params = {'max_depth': np.linspace(5, 10, 6, dtype=int)}#5
cv_params ={'min_child_weight':list(range(1,6,2)) }#5
cv_params = {'gamma': np.linspace(0, 0.1, 10)}#0
cv_params = {'reg_lambda': np.linspace(0, 100, 11)} #70
cv_params = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100,150,200]} #100
cv_params = {'subsample': np.linspace(0.5, 1, 6)}#1
cv_params = {'colsample_bytree': np.linspace(0.5, 1, 6)}#1
cv_params = {'learning_rate': np.linspace(0.04, 0.3, 11)} 

tscv = TimeSeriesSplit(n_splits=3)
params3 = {
 'learning_rate' : 0.11,
 'n_estimators' :70,
 'max_depth' : 4,
 'min_child_weight' : 2,
 'booster' : 'gbtree',
 'objective' : 'reg:squarederror',
 'colsample_bytree' : 1,
 'subsample' : 1,
 'gamma' : 0,
 'eval_metric' : ['mae','rmse'],
 'reg_lambda':0,
 'reg_alpha': 350}

regress_model = XGBRegressor(**params3) 
gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=tscv,scoring='neg_mean_absolute_error')
gs.fit(X3, y3) 
print("Optimal params:", gs.best_params_)

# After geting the hyperparameters selected using grid search, then train the model
tuned_gbr3 = xgb.XGBRegressor(**params3)
tuned_gbr3.fit(train3_X,train3_y, eval_set=[(train3_X,train3_y),(val3_X, val3_y)])



'''
test the models
'''

'''
use XGBOOSTs to classsification
'''

train1 = add_lags(train,48)
train1_cluster = train1.copy()
merged_data = train1_cluster.merge(daily_data[['datetime','cluster']], how='inner', on='datetime')
merged_data.dropna(inplace=True)
merged_data.shape
merged_data = merged_data.drop(['index','passengers','usage_kWh'],axis=1)


test.reset_index(inplace=True)
test_daily = test.pivot(index='datetime', columns='usage_timeID_zulu', values='usage_kWh')


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


daily_data1 = daily_data.copy()
cluster = daily_data1[['cluster']]
daily_data1 = daily_data1.drop(['cluster','datetime'],axis=1)
values =daily_data1.values

'''
use the usage of gas before one day to predict the cluster
'''

day_before_one = series_to_supervised(values,n_in=1,col_names=daily_data1.columns)
day_before_one.columns
day_before_one = day_before_one.iloc[:,:48]


#  Expected cluster names: [0 1 2], got [-1  0  1], so change rename noise cluster -1 into cluster 2
cluster[cluster == -1] = 2

cluster.reset_index(inplace=True)
cluster = cluster.drop('index',axis=1)
cluster = cluster.iloc[1:,:]


x_test = day_before_one.iloc[1000:,:]
y_test = cluster[1000:]
x_train1, y_train1, x_val1, y_val1 = split_train_val(day_before_one.iloc[:1000,:],cluster[:1000],0.8)
x_train1.info()


from xgboost.sklearn import XGBClassifier
'''
use the grid search to find the hyperparameters of XGBOOST classifier.
Choose one hyperparameter to be tuned for each time
'''

tscv = TimeSeriesSplit(n_splits=3)
cv_params = {'n_estimators': np.linspace(20, 50, 4, dtype=int)}#20
cv_params = {'max_depth': np.linspace(5, 10, 6, dtype=int)}#5
cv_params ={'min_child_weight':list(range(1,6,2)) }#5
cv_params = {'gamma': np.linspace(0, 0.1, 10)}#0
cv_params = {'lambda': np.linspace(0, 100, 11)} #70
cv_params = {'alpha':[1e-5, 1e-2, 0.1, 1, 100,150,200]} #100
cv_params = {'subsample': np.linspace(0.5, 1, 6)}#1
cv_params = {'colsample_bytree': np.linspace(0.5, 1, 6)}#1
cv_params = {'learning_rate': np.linspace(0.04, 0.3, 11)} 

params={'booster':'gbtree',
        'objective': 'multi:softmax',  
        'eval_metric': 'auc',
        'learning_rate' : 0.17,
        'n_estimators' :50,
        'max_depth' : 5,
        'min_child_weight' : 3,
        'colsample_bytree' : 1,
        'subsample' : 0.6,
        'gamma' : 0,
        'lambda':80,
        'alpha': 1}

model = XGBClassifier(**params) 
gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=tscv,scoring='neg_mean_absolute_error')
gs.fit(day_before_one, cluster) 
print("Optimal params:", gs.best_params_)

clf = XGBClassifier(**params)
clf.fit(x_train1, y_train1, eval_set=[(x_train1, y_train1),(x_val1, y_val1)])

from sklearn.metrics import roc_auc_score,roc_curve,auc

# evaluate the performance of xgboost classifier on test date set
y_pred_proba = clf.predict_proba(x_test)
auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # Use 'ovr' for multi-class problems
print("AUC Score:", auc_score)



# draw the ROC curve
n_classes = len(np.unique(y_test))
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (class %d) (AUC = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc="lower right")
plt.show()





'''
test the classification 
'''

test.reset_index(inplace=True)
test_daily = test.pivot(index='datetime', columns='usage_timeID_zulu', values='usage_kWh')


test_daily1 = test_daily.copy()
values = test_daily1.values
test_before_one = series_to_supervised(values,n_in=1,col_names=test_daily1.columns)
test_before_one.columns
test_before_one = test_before_one.iloc[:,:48] 

cluster_test = clf.predict(test_before_one) #Use the previous day's gas consumption to predict which cluster the next day belongs to
cluster_test.shape
test_daily1.dropna(inplace=True) #Deleted last day 6.13 since the last moment on 2023.6.13 is NA
test_daily1 = test_daily1.iloc[1:,:]
test_daily1['cluster'] = cluster_test
test_daily1.reset_index(inplace=True)


'''
test the two-stage model
'''


def Extract_test_data(cluster_num,lag,variables):
    
    '''
    Extrct the observations in each cluster
    cluster_num: the cluster number
    lag : the number of lags of gas consumption
    variables: Do not to be considered !
    '''
    
    test1 = add_lags(test,lag)
    test1_cluster = test1.copy()
    merged_data = test1_cluster.merge(test_daily1[['datetime','cluster']], how='inner', on='datetime')

    final_data = merged_data[merged_data['cluster']==cluster_num]
    X = final_data.dropna().drop(variables,axis=1)
    X.drop('index',axis =1,inplace=True)
    #X.drop('level_0',axis =1,inplace=True)
    y = final_data.dropna()['usage_kWh']
    
    return X, y

variables = ['datetime','usage_kWh','passengers','cluster']
variables2 = ['datetime','usage_kWh','atms','cluster']
variables3 = ['datetime','usage_kWh','atms','passengers','cluster']
variables4 = ['datetime','usage_kWh','atms','passengers','cluster','sunlight_duration_minutes']
variables5 = ['datetime','usage_kWh','atms','passengers','cluster','sunlight_duration_minutes','usage_dateID_zulu','usage_timeID_zulu']



# cluster1
X1_test,y1_test = Extract_test_data(1,24,variables3)
X1_test.info()


y_pre1 = tuned_gbr1.predict(X1_test)
mean_absolute_error(y1_test, y_pre1) # 69.6853265 (MAE)
mean_absolute_percentage_error(y1_test, y_pre1) #MAPE
np.sqrt(mean_squared_error(y1_test, y_pre1)) #RMSE

#cluster0
X2_test,y2_test = Extract_test_data(0,24,variables3)
X2_test.info()

y_pre2 = tuned_gbr2.predict(X2_test)
mean_absolute_error(y2_test, y_pre2) #81.74233 ，not general (MAE)
mean_absolute_percentage_error(y2_test, y_pre2)#MAPE
np.sqrt(mean_squared_error(y2_test, y_pre2)) #RMSE

# cluster -1(2)
X3_test,y3_test = Extract_test_data(2,48,variables3)
X3_test.info()

y_pre3 = tuned_gbr3.predict(X3_test)
mean_absolute_error(y3_test, y_pre3) #MAE
mean_absolute_percentage_error(y3_test, y_pre3)#MAPE
np.sqrt(mean_squared_error(y_pre3,y3_test))  #RMSE


