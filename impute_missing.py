#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 00:22:45 2023

@author: air
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from datetime import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

def date_parser(x):
    return pd.to_datetime(x, format="%d/%m/%Y %H:%M")
full_data = pd.read_csv('/Users/air/Desktop/processed_data.csv',date_parser=date_parser)
full_data.drop('Unnamed: 0',axis= 1, inplace=True)
full_data.columns
full_data.isna().sum()
full_data.info()

full_data['passengers'] = full_data['passengers'].str.replace(',', '').astype(int)
full_data['usage_dateID_zulu'] = pd.to_datetime(full_data['usage_dateID_zulu'],format='%Y/%m/%d').dt.strftime('%Y%m%d').astype(int)


X = full_data.dropna()[['sunlight_duration_minutes','usage_dateID_zulu', 'usage_timeID_zulu',
                        'passengers', 'atms']] 
y = full_data.dropna()['usage_kWh'] # target variable



mutual_infos = mutual_info_regression(X, y)
mutual_infos_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_infos})
mutual_infos_df.sort_values(by='Mutual Information', ascending=False, inplace=True)
threshold = 0.01
selected_features = mutual_infos_df[mutual_infos_df['Mutual Information'] > threshold]['Feature']

print('Selected Features:')
print(selected_features)

# import test data
test_data = pd.read_csv('/Users/air/Desktop/test_data.csv')
test_data.drop('Unnamed: 0',axis= 1, inplace=True)
test_data.isna().sum()
test_data['usage_dateID_zulu'] =pd.to_datetime(test_data['usage_dateID_zulu']).dt.strftime('%Y%m%d').astype(int)
test_data.info()


# combine full_data and test_data 
Data = pd.concat([full_data,test_data],axis = 0)
Data.isna().sum()

Data[['usage_dateID_zulu','sunlight_duration_minutes']].drop_duplicates()

#correlation 
correlation_matrix = Data.corr()

# Generate correlated heat maps
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# correlation test
corr_test = stats.pearsonr(Data.dropna()['atms'], Data.dropna()['usage_kWh'])
corr_coefficient, p_value = corr_test
print("Correlation coefficient:", corr_coefficient)
print("p-value:", p_value)


# check the multicollinearity
Data['intercept'] = 1
Data.dropna(inplace = True)
def calculate_vif(dataframe, target_column):
   X = dataframe.drop(columns=[target_column])
   y = dataframe[target_column]
   model = sm.OLS(y, X).fit()
   r_squared = model.rsquared
   vif = 1 / (1 - r_squared)
   return vif

# calculate the VIFs for each features
vif_values = {}
for column in Data.columns:
    vif_values[column] = calculate_vif(Data, column)


print("VIF Values:")
for column, vif in vif_values.items():
    print(f"{column}: {vif}")

# acf and pacf
''' 
show an obvious seasonal pattern, so the period = 48
'''

plot_pacf(Data['usage_kWh'], lags=48)  # has NA s
plot_acf(Data['usage_kWh'],lags=192) 







# imput sunlight_duration_minutes
Data1 = Data.copy()
Data1.isna().sum()
Data1.columns
Data1.info()

# check  the date that there is no sunlight_duration_minutes
missing_dates_sun = np.sort(Data1[Data1['sunlight_duration_minutes'].isnull()]['usage_dateID_zulu'].unique())
missing_dates_sun = pd.to_datetime(missing_dates_sun,format='%Y%m%d').strftime('%Y-%m-%d')


# using the sunlight_duration_minutes in last or next year to fill the missing 
Data1['usage_dateID_zulu'] = pd.to_datetime(Data1['usage_dateID_zulu'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
# set date as index
Data1.set_index('usage_dateID_zulu',inplace=True)

data_2020 = Data1[pd.to_datetime(Data1.index).year == 2020]
data_2021 = Data1[pd.to_datetime(Data1.index).year == 2021]
data_2022 = Data1[pd.to_datetime(Data1.index).year == 2022]


#data_2021[data_2021.index == '2021-08-01']

for i in missing_dates_sun:
    if pd.to_datetime(i).year == 2019 and pd.to_datetime(i).month != 8: # use 2020.02 10 to fill 2019.02 10
    
        target_date = pd.to_datetime(i) + relativedelta(years=1)
        target_date_str = target_date.strftime('%Y-%m-%d')
        Data1.loc[i, 'sunlight_duration_minutes'] = data_2020.loc[target_date_str, 'sunlight_duration_minutes'].values
        
    elif pd.to_datetime(i).year == 2019 and pd.to_datetime(i).month == 8:# use 2021.08 to fill 2019.08
        target_date = pd.to_datetime(i) + relativedelta(years=2)
        target_date_str = target_date.strftime('%Y-%m-%d')
        Data1.loc[i, 'sunlight_duration_minutes'] = data_2021.loc[target_date_str, 'sunlight_duration_minutes'].values
    
    elif pd.to_datetime(i).year == 2020: # use 2021.08 to fill 2020.08
        target_date = pd.to_datetime(i) + relativedelta(years=1)
        target_date_str = target_date.strftime('%Y-%m-%d')
        Data1.loc[i, 'sunlight_duration_minutes'] = data_2021.loc[target_date_str, 'sunlight_duration_minutes'].values
        
    elif pd.to_datetime(i).year == 2021: # use 2022.10 to fill 2021.10
        target_date = pd.to_datetime(i) + relativedelta(years=1)
        target_date_str = target_date.strftime('%Y-%m-%d')
        Data1.loc[i, 'sunlight_duration_minutes'] = data_2022.loc[target_date_str, 'sunlight_duration_minutes'].values
        
    elif pd.to_datetime(i).year == 2023: # use 2022 to fill 2023
        target_date = pd.to_datetime(i) - relativedelta(years=1)
        target_date_str = target_date.strftime('%Y-%m-%d')
        Data1.loc[i, 'sunlight_duration_minutes'] = data_2022.loc[target_date_str, 'sunlight_duration_minutes'].values
        
#check results of imputing in sunlight_duration_minutes
Data1[Data1['sunlight_duration_minutes'].isnull()].index.unique()
Data1.reset_index(inplace=True)
Data1['usage_dateID_zulu'] = pd.to_datetime(Data1['usage_dateID_zulu']).dt.strftime('%Y%m%d').astype(int)
Data1.info()

#imput usage_kWh 

Ytrain = Data1[Data1['usage_kWh'].notnull()]['usage_kWh']
Xtrain =  Data1[Data1['usage_kWh'].notnull()][['usage_dateID_zulu', 'usage_timeID_zulu', 'sunlight_duration_minutes','passengers', 'atms']]


Xtest = Data1[Data1['usage_kWh'].isnull()][['usage_dateID_zulu', 'usage_timeID_zulu', 'sunlight_duration_minutes','passengers', 'atms']]


# use random forest to impute Na in usage_kWh
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

X_train, X_val, y_train, y_val = train_test_split(Xtrain,Ytrain, test_size=0.8, random_state=2)

rfc = RandomForestRegressor(n_estimators=100)
rfc = rfc.fit(X_train, y_train)
y_pred =  rfc.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error (MAE):", mae)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_missing_pred = reg_model.predict(X_val)
mean_absolute_error(y_val, y_missing_pred)

'''
compare mae of linear and RF , RF has smaller MAE,so select MAE to impute
'''
rfc = RandomForestRegressor(n_estimators=100)
rfc = rfc.fit(Xtrain, Ytrain)

Data2 = Data1.copy()
imputed_values = rfc.predict(Xtest)
Data2.loc[Data2['usage_kWh'].isnull(), 'usage_kWh'] = imputed_values
Data2.isna().sum()

'''
Data2 is the gas and atm data combined after using RF to impute NA values:sunlight_duration_minutes,usage_kWh
'''
#Data2.to_csv('/Users/air/Desktop/imputed_total_data.csv')





