import zipfile
import shutil
import pandas as pd

def extract_zip(file):
    # Unzip a file
    zip_ref = zipfile.ZipFile(file, 'r')
    folder = '/'.join(file.split('/')[:-1])
    zip_ref.extractall(folder)
    zip_ref.close()

def load_files(file_name, variables, farm_number):
    dfs = []
    for i in range(1,farm_number+1):
        file = file_name.format(i)
        df = pd.read_csv(file, index_col=1, parse_dates=True)
        df = df.pivot_table(index='TIMESTAMP', columns=['ZONEID'], values=variables, dropna=False)
        df.columns = df.columns.swaplevel(i=0, j=1)
        dfs.append(df)
    df_tasks = pd.concat(dfs, axis=1)

    return df_tasks

def load_wind_track():

    # Unzip files
    extract_zip('./data/gefcom2014/GEFCom2014 Data/GEFCom2014-W_V2.zip')
    extract_zip('./data/gefcom2014/GEFCom2014 Data/Wind/Task 15/Task15_W_Zone1_10.zip')
    extract_zip('./data/gefcom2014/GEFCom2014 Data/Wind/Task 15/TaskExpVars15_W_Zone1_10.zip')

    # Get all target data except for last task
    df_task1_14 = load_files('./data/gefcom2014/GEFCom2014 Data/Wind/Task 15/Task15_W_Zone1_10/Task15_W_Zone{0}.csv', ['TARGETVAR', 'U10', 'V10', 'U100','V100'], 10)

    # Get explanatory variables data for all task
    df_exp15 = load_files('./data/gefcom2014/GEFCom2014 Data/Wind/Task 15/TaskExpVars15_W_Zone1_10/TaskExpVars15_W_Zone{0}.csv', ['U10', 'V10', 'U100','V100'], 10)

    # Get target data for last task
    df_target15 = pd.read_csv('./data/gefcom2014/GEFCom2014 Data/Wind/Solution to Task 15/solution15_W.csv', index_col=1, parse_dates=True)
    df_target15 = df_target15.pivot_table(index='TIMESTAMP', columns=['ZONEID'], values=['TARGETVAR'], dropna=False)
    df_target15.columns = df_target15.columns.swaplevel(i=0, j=1)

    df_task15 = pd.merge(df_target15, df_exp15, on='TIMESTAMP')
    df_tasks = pd.concat([df_task1_14, df_task15], axis=0)
    df_tasks.to_csv('./data/gefcom2014/raw/gefcom2014-wind-raw.csv')
    print('Wind track data saved to: ./data/raw/gefcom2014-wind-raw.csv.')

def load_solar_track():

    # Unzip files
    extract_zip('./data/gefcom2014/GEFCom2014 Data/GEFCom2014-S_V2.zip')

    # Get all explanatory and target data except for all tasks
    df = pd.read_csv('./data/gefcom2014/GEFCom2014 Data/Solar/Task 15/predictors15.csv', header=0, index_col=1, parse_dates=True)
    df = df.pivot_table(index='TIMESTAMP', columns=['ZONEID'], values=['POWER', 'VAR78', 'VAR79', 'VAR134', 'VAR157', 'VAR164', 'VAR165', 'VAR166', 'VAR167', 'VAR169', 'VAR175', 'VAR178', 'VAR228'], dropna=False)
    df.columns = df.columns.swaplevel(i=0, j=1)
    df.to_csv('./data/gefcom2014/raw/gefcom2014-solar-raw.csv')
    print('Solar track data saved to: ./data/raw/gefcom2014-solar-raw.csv.')

def load_load_track():

    # Unzip files
    extract_zip('./data/gefcom2014/GEFCom2014 Data/GEFCom2014-L_V2.zip')

    dfs = []
    for task in range(1,16):
        file = './data/gefcom2014/GEFCom2014 Data/Load/Task {0}/L{0}-train.csv'.format(task)
        df = pd.read_csv(file, header=0)
        dfs.append(df)
    df = pd.concat(dfs)

    # The dates are ambiguous so need to hardcode them. 
    df = df.drop(columns=['TIMESTAMP', 'ZONEID'])
    index = pd.date_range(start='2001-01-01 01:00', end='2011-12-01 00:00', freq='H')
    df.index = index
    df.index.name = 'datetime'

    df_task15 = pd.read_csv('./data/gefcom2014/GEFCom2014 Data/Load/Solution to Task 15/solution15_L_temperature.csv')
    df_task15.index = pd.to_datetime(df_task15['date'])+pd.to_timedelta(df_task15['hour'], unit='h')
    df_task15.index.name = 'datetime'
    df_task15 = df_task15.drop(columns=['date', 'hour'])
    df = pd.concat([df, df_task15])

    df.to_csv('./data/gefcom2014/raw/gefcom2014-load-raw.csv')
    print('Load track data saved to: ./data/raw/gefcom2014-load-raw.csv.')

if __name__ == '__main__':
    extract_zip('./data/gefcom2014/1-s2.0-S0169207016000133-mmc1.zip')
    load_wind_track()
    load_solar_track()
    load_load_track()
    shutil.rmtree('./data/gefcom2014/GEFCom2014 Data')
