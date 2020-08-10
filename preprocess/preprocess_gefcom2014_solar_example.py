#!/usr/bin/python

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

sys.path.append('../')
from ranewable.ranewable import Ra

def load_data(path_data, header=0):

    df = pd.read_csv(path_data, header=header, index_col=0, parse_dates=True)

    return df


def preprocess_data(df, params_json):

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

    # Average point features over hour
    features_point = ['VAR134', 'VAR157', 'VAR164', 'VAR165', 'VAR166', 'VAR167', 'VAR78', 'VAR79']
    df_point = df.loc[:,(slice(None),features_point)]
    df_point = df_point.rolling(2).mean().shift(-1).fillna(method='ffill')
    df.loc[:,(slice(None),features_point)] = df_point

    # Differentiate accumulated features
    features_accum = ['VAR169', 'VAR175', 'VAR178', 'VAR228']
    df_accum = df.loc[:,(slice(None),features_accum)]
    df_accum = df_accum.diff()
    df_accum[df_accum.index.levels[1].hour==1] = df.loc[df_accum.index.levels[1].hour==1,(slice(None),features_accum)]
    df_accum.loc[:,(slice(None),features_accum[:3])] = df_accum.loc[:,(slice(None),features_accum[:3])]/3600 # Convert from J to Wh/h
    df.loc[:,(slice(None),features_accum)] = df_accum

    # Add lead time feature
    ref_datetime = df.index.get_level_values(0)
    valid_datetime = df.index.get_level_values(1)
    lead_time = (valid_datetime-ref_datetime)/pd.Timedelta('1 hour')
    for farm in df.columns.levels[0]:
        df.loc[:,(farm,'LEAD_TIME')] = lead_time

    # Add ranewable features
    for i, (coords, alt, cap, orien, tilt) in enumerate(zip(params_json['farm_coords'],
                                                            params_json['farm_altitude'],
                                                            params_json['farm_capacity'],
                                                            params_json['panel_orientation'],
                                                            params_json['panel_tilt'])):
        ra =  Ra(longitude=coords[0],
                 latitude=coords[1],
                 altitude=alt,
                 capacity=cap,
                 orientation=orien,
                 tilt=tilt)

        df_solpos = ra.calculate_solpos(df[str(i+1)].index)
        df_clearsky = ra.calculate_clearsky(df[str(i+1)].index)
        df_power_clearsky = ra.calculate_power_clearsky(df[str(i+1)].index)
        df_weather = ra.weather_from_ghi(df.loc[:,(str(i+1),'VAR169')])
        df_power = ra.calculate_power(df_weather.copy())

        df_solpos = df_solpos.loc[:, ['zenith', 'azimuth']]
        df_clearsky.columns = df_clearsky.columns+'_clearsky'
        df_weather = df_weather.loc[:, ['dni', 'dhi', 'ghi', 'kt']]

        for column in df_solpos.columns:
            df.loc[:,(str(i+1),column)] = df_solpos.loc[:, column]
        for column in df_clearsky.columns:
            df.loc[:,(str(i+1),column)] = df_clearsky.loc[:, column]
        for column in df_weather.columns:
            df.loc[:,(str(i+1),column)] = df_weather.loc[:, column]
        df.loc[:,(str(i+1),'Clearsky_Forecast')] = df_power_clearsky
        df.loc[:,(str(i+1),'Physical_Forecast')] = df_power

    # Difference between real power and physical forecast
    for farm in df.columns.levels[0]:
        df.loc[:,(farm,'DIFF')] = (df.loc[:,(farm,'POWER')]-df.loc[:,(farm,'Physical_Forecast')])


    return df


def save_data(path, df):
    
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path+'gefcom2014-solar-preprocessed.csv')


if __name__ == '__main__':
    params_path = sys.argv[1]
    with open(params_path, 'r', encoding='utf-8') as file:
        params_json = json.loads(file.read())

    df = load_data(params_json['path_raw_data'], header=[0,1])
    df = preprocess_data(df, params_json)
    save_data(params_json['path_preprocessed_data'], df)
    print('Solar track preprocessed data saved to: '+params_json['path_preprocessed_data'])
