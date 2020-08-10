#!/usr/bin/python

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm


sys.path.append('../')

def load_data(path_data, header=0):

    df = pd.read_csv(path_data, header=header, index_col=0, parse_dates=True)

    return df


def preprocess_data(df,params_json):
    
    df['hour'] = df.index.hour.astype('category')
    df['weekday'] = df.index.weekday.astype('category')
    df['month'] = df.index.month.astype('category')

    # Convert to standard indexing structure (ref_datetime, valid_datetime)
    df.index.name = 'valid_datetime'
    idx_ref_datetime = (df.index.hour == 1) & (df.index.day == 1)
    df.loc[idx_ref_datetime, 'ref_datetime'] = df.index[idx_ref_datetime]
    df.loc[:, 'ref_datetime'] = df.loc[:, 'ref_datetime'].fillna(method='ffill')
    df = df.set_index('ref_datetime', append=True, drop=True)
    df.index = df.index.reorder_levels(['ref_datetime', 'valid_datetime'])
    df = df.sort_index()
    
    # Average temperature only
    #df['TEMP'] = df.drop(columns='LOAD').mean(axis=1)

    datetimes = df.index.get_level_values(1)

    stations = ['w6', 'w10', 'w22', 'w25']
    df_train = df.loc[pd.IndexSlice[:, :'2010-09-01 01:00:00'], stations]
    valid_datetimes_train = df_train.index.get_level_values(1)
    means = np.empty((len(datetimes),4))
    stds = np.empty((len(datetimes),4))
    with tqdm(total=len(datetimes)) as pbar:
       for i, datetime  in enumerate(datetimes):
           hour = datetime.hour
           dayofyear = datetime.dayofyear

           similar_days = (valid_datetimes_train.hour == hour) & (np.abs(valid_datetimes_train.dayofyear-dayofyear) < 5)
           means[i,:] = df_train[similar_days].mean().values
           stds[i,:] = df_train[similar_days].mean().values
            
           pbar.update(1)
    
    for i, station in enumerate(stations):         
       df.loc[:, station+'_m'] = means[:,i]
       df.loc[:, station+'_s'] = stds[:,i]
    
    # Drop columns
    drop_cols = ['w'+str(i) for i in range(1,26)]
    df = df.drop(columns=drop_cols)
    df = df.dropna()

    # Add multicolumn according to standard site_forecast format
    df.columns = pd.MultiIndex.from_product([['1'], df.columns])
    
    return df

def save_data(path, df):
    
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path+'gefcom2014-load-preprocessed.csv')


if __name__ == '__main__':
    params_path = sys.argv[1]
    with open(params_path, 'r', encoding='utf-8') as file:
        params_json = json.loads(file.read())

    df = load_data(params_json['path_raw_data'])
    df = preprocess_data(df, params_json)
    save_data(params_json['path_preprocessed_data'], df)
    print('Load track preprocessed data saved to: '+params_json['path_preprocessed_data'])
