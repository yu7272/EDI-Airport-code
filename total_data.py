#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 23:41:56 2023

@author: air
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

#Load data
date_parser = lambda x: pd.datetime.strptime(x, "%d/%m/%Y %H:%M")
original_gas = pd.read_csv('/Users/air/Desktop/Dissertation /code/gas_consumption.csv',parse_dates=['usage_date_time'], date_parser=date_parser)
original_atm = pd.read_csv('/Users/air/Desktop/Dissertation /code/pax_atms.csv',parse_dates=['usage_date_time'], date_parser=date_parser)



# fill the NA in atm
original_atm = original_atm.fillna(0)
atm = original_atm.copy()
atm.set_index('usage_date_time', inplace=True)


original_gas['usage_kWh'] = original_gas['usage_kWh'].str.replace(',', '').astype(float)
original_gas['usage_dateID_zulu'] = original_gas['usage_dateID_zulu'].replace(',', '').astype(int)
original_gas['usage_timeID_zulu'] = original_gas['usage_timeID_zulu'].replace(',', '').astype(int)
original_gas['sunlight_duration_minutes'] = original_gas['sunlight_duration_minutes'].str.replace(',', '').astype(float)

# don't consider temperature and weather_observation_timeID_local
gas = original_gas[['usage_dateID_zulu','usage_timeID_zulu','sunlight_duration_minutes','usage_kWh']]
complete_datetime = pd.date_range(start='2019-01-01 00:00:00', end='2023-06-13 23:30:00',freq='30T')
date = complete_datetime.strftime('%Y%m%d')
time = complete_datetime.strftime('%H%M%S')

# delete the data in 2023 which will be used as a test set
gas = gas[pd.to_datetime(gas['usage_dateID_zulu'].astype(str), format='%Y%m%d').dt.year != 2023]


gas['usage_dateID_zulu'].value_counts()
#calculate value_counts
counts = gas['usage_dateID_zulu'].value_counts()
counts.unique() #([50, 48, 46,  2])
len(counts) #1332
len(counts[counts == 48].index) # 1318
len(counts[counts == 50].index) #3 delete first 2 records
len(counts[counts == 46].index) #8 add last 2 records
len(counts[counts == 2].index)#3 

gas1 = gas.copy(deep=True)
# delete first 2 records
for i in counts[counts == 50].index:
    gas1 = gas1.drop(gas1[gas1['usage_dateID_zulu'] == i].head(2).index)

# add last 2 records
gas2 = gas1.copy()
index_back = [20200731,20210930]
#j = 20200731
for j in index_back:
    ind = gas2[gas2['usage_dateID_zulu'] == j].tail(1).index.item()
    sun_value = gas2.iloc[ind-1]['sunlight_duration_minutes']
    gas_vaue = np.mean(gas2.iloc[ind-1:ind+1]['usage_kWh']) 
    
    df1 = gas2.iloc[:ind+1,]
    added_row = pd.DataFrame({'usage_dateID_zulu': [j,j], 'usage_timeID_zulu': [232959,235959],
                              'sunlight_duration_minutes':[sun_value,sun_value],'usage_kWh':[gas_vaue,gas_vaue]})
    added_row.index = [ind+1,ind+2]
    df2 = gas2.iloc[ind+1:,]
    df2.index += 2
    
    gas2 = pd.concat([df1, added_row, df2])

# add last 3 records
gas3 = gas2.copy()
index_back = [20190731, 20190930]
#j = 20200731
for j in index_back:
    ind = gas3[gas3['usage_dateID_zulu'] == j].tail(1).index.item()
    sun_value = gas3.iloc[ind-1]['sunlight_duration_minutes']
    gas_vaue = np.mean(gas3.iloc[ind-1:ind+1]['usage_kWh']) 
    
    df1 = gas3.iloc[:ind+1,]
    added_row = pd.DataFrame({'usage_dateID_zulu': [j,j,j], 'usage_timeID_zulu': [225959,232959,235959],
                              'sunlight_duration_minutes':[sun_value,sun_value,sun_value],'usage_kWh':[gas_vaue,gas_vaue,gas_vaue]})
    added_row.index = [ind+1,ind+2,ind+3]
    df2 = gas3.iloc[ind+1:,]
    df2.index += 3
    
    gas3 = pd.concat([df1, added_row, df2])


# add first 2 records
gas4 = gas3.copy()
index_head = [20211031, 20221030, 20201025]
#j = 20200731
for j in index_head:
    ind = gas4[gas4['usage_dateID_zulu'] == j].head(1).index.item()
    sun_value = gas4.iloc[ind]['sunlight_duration_minutes']
    gas_vaue = np.mean(gas4.iloc[ind-1:ind+1]['usage_kWh']) 
    
    df1 = gas4.iloc[:ind-1,]
    added_row = pd.DataFrame({'usage_dateID_zulu': [j,j], 'usage_timeID_zulu': [2959,5959],
                              'sunlight_duration_minutes':[sun_value,sun_value],'usage_kWh':[gas_vaue,gas_vaue]})
    added_row.index = [ind+1-2,ind+2-2]
    df2 = gas4.iloc[ind-1:,]
    df2.index += 2
    
    gas4 = pd.concat([df1, added_row, df2])
    
    
# add records on 20190331
gas5 = gas4.copy()
ind = gas5[gas5['usage_dateID_zulu'] == 20190331].iloc[1].name
sun_value = gas5.iloc[ind]['sunlight_duration_minutes']
gas_vaue = np.mean(gas4.iloc[ind-1:ind+1]['usage_kWh']) 
df1 = gas5.iloc[:ind+1,]
added_row = pd.DataFrame({'usage_dateID_zulu': [20190331,20190331], 'usage_timeID_zulu': [5959,12959],
                          'sunlight_duration_minutes':[sun_value,sun_value],'usage_kWh':[gas_vaue,gas_vaue]})
added_row.index = [ind+1,ind+2]
df2 = gas5.iloc[ind+1:,]
df2.index += 2

gas5 = pd.concat([df1, added_row, df2])

gas5['usage_dateID_zulu'].value_counts().unique()


# change index into usage_dateID_zulu and usage_timeID_zulu
gas6= gas5.copy()
gas6['usage_dateID_zulu'] = pd.to_datetime(gas6['usage_dateID_zulu'],format='%Y%m%d').astype(str)
gas6.set_index(['usage_dateID_zulu','usage_timeID_zulu'], inplace=True)
gas6.index



# move back 30 minutes
new_index = pd.MultiIndex.from_tuples([ ('2020-02-29', 235959)])
data_gas = gas6.append(pd.DataFrame(index=new_index)).sort_index()
data_gas.index
data_gas.drop(index=pd.MultiIndex.from_tuples([('2019-01-01',      0)]),inplace=True)
data_gas.loc[('2020-02-29', 235959)]
data_gas.loc[('2019-07-31', 235959)]

#reset index
data_gas = data_gas.reset_index()

# IF usage_timeID_zulu = 0, then the date - 1 day 
data_gas['usage_dateID_zulu'] = pd.to_datetime(data_gas['usage_dateID_zulu'], format='%Y-%m-%d')
data_gas.loc[data_gas['usage_timeID_zulu'] == 0, 'usage_dateID_zulu'] -= pd.Timedelta(days=1)
data_gas.loc[data_gas['usage_timeID_zulu'] == 0, 'sunlight_duration_minutes'] =  data_gas.loc[data_gas.loc[data_gas['usage_timeID_zulu'] == 0].index-1]['sunlight_duration_minutes'].values
data_gas.loc[data_gas['usage_timeID_zulu'] == 0, 'usage_timeID_zulu'] = 235959

data_gas.loc[data_gas['usage_dateID_zulu']=='2019-01-02' ]

x = data_gas['usage_dateID_zulu'].value_counts()
x[x!=48]


# change index into usage_dateID_zulu and usage_timeID_zulu
data_gas1= data_gas.copy()
data_gas1['usage_dateID_zulu'] = pd.to_datetime(data_gas1['usage_dateID_zulu'],format='%Y%m%d').astype(str)
data_gas1.set_index(['usage_dateID_zulu','usage_timeID_zulu'], inplace=True)
data_gas1.index


#data_gas[data_gas['usage_dateID_zulu'] == '2019-07-31'].shape





#change corresponding usage_timeID_zulu
start_time = '00:29:59'
end_time = '23:59:59'
interval = '30min'
times = pd.date_range(start=start_time, end=end_time, freq=interval).strftime('%H%M%S').astype(int).tolist()
#data_gas.index.get_level_values(0).value_counts()

#new_multi_index = data_gas.index.set_levels(times, level=1)

# impute the NA months in 2019 

def insert_index(start, end,data):
    new_index = pd.MultiIndex.from_tuples([(i, j) for i in pd.date_range(start,end,freq='D').strftime('%Y-%m-%d') for j in times])
    data = data.append(pd.DataFrame(index=new_index))
    return data

data_gas2 = data_gas1.copy()
data_gas2 = insert_index('2019-02-01', '2019-02-28', data_gas2)
data_gas2.index[data_gas2.index.get_level_values(0)=='2019-02-28']
data_gas2.loc[('2019-02-28', 235959)]
data_gas2.loc[('2019-02-28', 235959)] = data_gas2.loc[('2019-02-28', 235959)].dropna()



data_gas2 = insert_index('2019-08-01', '2019-08-31',data_gas2)
data_gas2.index[data_gas2.index.get_level_values(0)=='2019-08-31']
data_gas2.loc[('2019-08-31', 235959)] = data_gas2.loc[('2019-08-31', 235959)].dropna()

data_gas2 = data_gas2.drop(index=[('2019-08-31', 225959),('2019-08-31', 232959)])

data_gas2.index

data_gas2 = insert_index('2019-10-01', '2019-10-31',data_gas2)
data_gas2.index[data_gas2.index.get_level_values(0)=='2019-10-31']
data_gas2.loc[('2019-10-31', 235959)] = data_gas2.loc[('2019-10-31', 235959)].dropna()

# On 2019-01-31, the last record is missing
ii = pd.MultiIndex.from_tuples([('2019-01-31', 235959)])
data_gas2 = data_gas2.append(pd.DataFrame(index=ii))

# for 2020
data_gas2 = insert_index('2020-08-01', '2020-08-31',data_gas2)
data_gas2.index[data_gas2.index.get_level_values(0)=='2020-08-31']
data_gas2.loc[('2020-08-31', 232959)] = data_gas2.loc[('2020-08-31', 232959)].dropna()
data_gas2.loc[('2020-08-31', 235959)] = data_gas2.loc[('2020-08-31', 235959)].dropna()


#for 2021
data_gas2 = insert_index('2021-10-01', '2021-10-11',data_gas2)
data_gas2.index[data_gas2.index.get_level_values(0)=='2021-10-11']
data_gas2.loc[('2021-10-11', 232959)] = data_gas2.loc[('2021-10-11', 232959)].dropna()
data_gas2.loc[('2021-10-11', 235959)] = data_gas2.loc[('2021-10-11', 235959)].dropna()




# order by time
data_gas2 = data_gas2.sort_index()

#reindex
data_gas3 = data_gas2.reset_index()

#check the duplicates terms
data_gas3[data_gas3.duplicated()]
#delete the duplicates
data_gas3.drop_duplicates(inplace=True) #71087


x1 = data_gas3['usage_dateID_zulu'].value_counts()
x1[x1!=48]

data_gas3[data_gas3['usage_dateID_zulu']=='2019-01-31']


#check the missing dates

def missing_dates(start_date, end_date, data):
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    existing_dates = pd.to_datetime(data['usage_dateID_zulu']).dt.date
    missing_dates = all_dates[~all_dates.isin(existing_dates)]
    missing_dates_df = pd.DataFrame({'dates': missing_dates})
    return missing_dates_df

    
#check the data of atm from 2019 to 2022
atm.loc['2019-01-01 00:00:00':'2022-12-31 23:30:00'].shape #(70128, 2)
    

    
# set usage_dateID_zulu asindex
data_gas3.columns   
data_gas3.set_index('usage_dateID_zulu', inplace=True)
data_gas3.index

# let time starts at 29:59
all_dates = pd.date_range(start='2019-01-01', end='2022-12-31', freq='D').strftime('%Y-%m-%d')
for k in all_dates:
    data_gas3.loc[k]['usage_timeID_zulu'] = times 

#delete the data of atms in 2023
atm1 = atm['2019-01-01 00:30:00':'2023-01-01 00:00:00']

'''
data_gas3 is the final processed gas data, atm1 is the final processed atm data 
from 2019 to 2022
'''

#combine two data
final_data = data_gas3.copy()

atm1.reset_index(inplace = True)
final_data.reset_index(inplace = True)

final_data['passengers'] = atm1['passengers']
final_data['atms'] = atm1['atms']

#final_data.to_csv('/Users/air/Desktop/processed_data.csv')
'''
final_data is the combination of gas_consumption and atms from 2019 to 2022
'''


# 2023: test data
gas = original_gas[['usage_dateID_zulu','usage_timeID_zulu','sunlight_duration_minutes','usage_kWh']]
gas2023 = gas[pd.to_datetime(gas['usage_dateID_zulu'],format='%Y%m%d').dt.year == 2023]
gas2023['usage_dateID_zulu'] = pd.to_datetime(gas2023['usage_dateID_zulu'],format='%Y%m%d')
gas2023.shape #(7824, 4)

atm2 = atm['2023-01-01 00:30:00':]
atm2.shape    # (7871, 2)
atm2.reset_index(inplace=True)

#check NA
count_2023 = gas2023['usage_dateID_zulu'].value_counts()
count_2023.unique() #array([50, 48, 46])

count_2023[count_2023==50].index

# delete first 2 records
for i in count_2023[count_2023==50].index:
    gas2023 = gas2023.drop(gas2023[gas2023['usage_dateID_zulu'] == i].head(2).index)

# add last two records
count_2023[count_2023==46].index #'2023-06-13'

ind = gas2023[gas2023['usage_dateID_zulu'] == '2023-06-13'].tail(1).index.item()
sun_value = gas2023.loc[ind]['sunlight_duration_minutes']
gas_vaue = gas2023.loc[ind]['usage_kWh'] 

added_row = pd.DataFrame({'usage_dateID_zulu': pd.to_datetime('20230613',format='%Y-%m-%d'), 'usage_timeID_zulu': [232959],
                          'sunlight_duration_minutes':[sun_value],'usage_kWh':[gas_vaue]})
added_row.index = [ind+1]

gas2023_1 = pd.concat([gas2023, added_row])
gas2023_1['usage_dateID_zulu'] = pd.to_datetime(gas2023_1['usage_dateID_zulu'], format='%Y-%m-%d')
gas2023_1.shape

#find the missing dates in 2023
miss_dates = missing_dates('2023-01-01', '2023-06-13', gas2023_1)


#fill the NA for missing dates
gas2023_1.set_index(['usage_dateID_zulu','usage_timeID_zulu'],inplace=True)
gas2023_1.index

gas2013_2 =insert_index('2023-01-04', '2023-01-04',  gas2023_1)
gas2013_2.shape # (7871, 4)
gas2013_2.reset_index(inplace = True)

#Sort the inserted date
gas2013_2.set_index(['usage_dateID_zulu','usage_timeID_zulu'],inplace=True)
gas2013_2.sort_index(inplace=True)
gas2013_2.reset_index(inplace = True)


#combine gas and atm data in 2023: gas2013_2,atm2
final_2023 = gas2013_2.copy()
atm2['passengers'] = atm2['passengers'].str.replace(',', '').astype(float)
final_2023['passengers'] = atm2['passengers'].fillna(0)
final_2023['atms'] = atm2['atms']
'''
final_2023 是2023年总和的data
'''

#final_2023.to_csv('/Users/air/Desktop/test_data.csv')
final_2023.info()






correlation_matrix = final_data.corr()

# Generate correlated heat maps
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

corr_test = stats.pearsonr(final_data.dropna()['atms'], final_data.dropna()['usage_kWh'])
corr_coefficient, p_value = corr_test
print("Correlation coefficient:", corr_coefficient)
print("p-value:", p_value)






