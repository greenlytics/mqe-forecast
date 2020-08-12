#!/usr/bin/python

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd


def load_data(path, filename, header=0):

    df = pd.read_csv(path+filename, header=header, index_col=0, parse_dates=True)

    return df


def preprocess_wind(df, target, features):

    # Convert to standard indexing structure (ref_datetime, valid_datetime)
    df.index.name = 'valid_datetime'
    idx_ref_datetime = df.index.hour == 1
    df.loc[idx_ref_datetime, 'ref_datetime'] = df.index[idx_ref_datetime]
    df.loc[:, 'ref_datetime'] = df.loc[:, 'ref_datetime'].fillna(method='ffill')
    df = df.set_index('ref_datetime', append=True, drop=True)[df.columns.levels[0][:-1]]
    df.index = df.index.reorder_levels(['ref_datetime', 'valid_datetime'])
    df = df.sort_index()

    # Remove hidden ref_datetime column from multiindex
    columns = [df.columns.levels[0][:-1].values, df.columns.levels[1][:-1].values]
    df.columns = pd.MultiIndex.from_product(columns)

    for farm in df.columns.levels[0]:
        df_features = pd.DataFrame(index=df.index)

        df_features.loc[:,'Utot10'] = np.sqrt(df.loc[:,(farm, 'U10')]**2+df.loc[:,(farm, 'V10')]**2)
        df_features.loc[:,'Theta10'] = np.angle(df.loc[:,(farm, 'U10')]+df.loc[:,(farm, 'V10')]*1j, deg=True)
        df_features.loc[:,'Utot100'] = np.sqrt(df.loc[:,(farm, 'U100')]**2+df.loc[:,(farm, 'V100')]**2)
        df_features.loc[:,'Theta100'] = np.angle(df.loc[:,(farm, 'U100')]+df.loc[:,(farm, 'V100')]*1j, deg=True)

        df_features.loc[:,'Utot310'] = df_features.loc[:,'Utot10']**3
        df_features.loc[:,'Utot3100'] = df_features.loc[:,'Utot100']**3
        df_features.loc[:,'Utotdiff'] = df_features.loc[:,'Utot100']-df_features.loc[:,'Utot10']

        for feature in df_features.columns:
            df.loc[:, (farm, feature)] = df_features[feature]

    df_temp = pd.DataFrame(index=df.index, columns=pd.MultiIndex.from_product([df.columns.levels[0], target+features]))
    df_temp.loc[:, (slice(None), target+features)] = df.loc[:, (slice(None), target+features)]

    return df_temp


def save_data(path, filename, df):
    
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path+filename)


if __name__ == '__main__':
    params_path = sys.argv[1]
    with open(params_path, 'r', encoding='utf-8') as file:
        params_json = json.loads(file.read())

    df = load_data(params_json['path_raw_data'], params_json['filename_raw_data'], header=[0,1])
    df = preprocess_wind(df, params_json['target'], params_json['features'])
    save_data(params_json['path_preprocessed_data'], params_json['filename_preprocessed_data'], df)
    print('Wind track preprocessed data saved to: '+params_json['path_preprocessed_data']+params_json['filename_preprocessed_data'])

