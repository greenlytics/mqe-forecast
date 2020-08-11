#!/usr/bin/python

import sys
import os
import shutil
import json
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import catboost as cb
from sklearn import ensemble
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb

def load_data(path_data):
    # Load preprocessed data

    df = pd.read_csv(path_data, header=[0,1], index_col=[0,1], parse_dates=True)

    return df

class Trial:
    def __init__(self, params_json):
        # Mandatory input variables
        self.trial_name = params_json['trial_name']
        self.trial_comment = params_json['trial_comment']
        self.path_result = params_json['path_result']
        self.path_preprocessed_data = params_json['path_preprocessed_data']
        self.data_resolution = params_json['data_resolution']
        self.splits = params_json['splits']
        self.sites = params_json['sites']
        self.features = params_json['features']
        self.target = params_json['target']
        self.model_params = params_json['model_params']
        self.regression_params = params_json['regression_params']
        self.save_options = params_json['save_options']

        # Optional input variables
        if 'variables_lags' in params_json:
            self.variables_lags = params_json['variables_lags']
        else: 
            self.variables_lags = None
        if 'diff_target_with_physical' in params_json:
            self.diff_target_with_physical = params_json['diff_target_with_physical']
        else: 
            self.diff_target_with_physical = False
        if 'target_smoothing_window' in params_json:
            self.target_smoothing_window = params_json['target_smoothing_window']
        else: 
            self.target_smoothing_window = 1
        if 'train_on_day_only' in params_json:
            self.train_on_day_only = params_json['train_on_day_only']
        else: 
            self.train_on_day_only = False
        if 'weight_params' in params_json: 
            self.weight_params = params_json['weight_params']


    def generate_dataset(self, df, split, site): 

        def add_lags(df, variables_lags): 
            # Lagged features
            for variable, lags in variables_lags.items():
                for lag in lags:
                    df.loc[:, variable+'_lag{0}'.format(lag)] = df.loc[:, variable].groupby('ref_datetime').shift(lag)

            return df

        # Make target into list if not already
        self.target = [self.target] if isinstance(self.target, str) else self.target

        if self.diff_target_with_physical and not (['Physical_Forecast'] in self.features):
            df_X = df[site].loc[pd.IndexSlice[:, split[0]:split[1]], self.features+['Physical_Forecast']]
        else:
            df_X = df[site].loc[pd.IndexSlice[:, split[0]:split[1]], self.features]

        df_y = df[site].loc[pd.IndexSlice[:, split[0]:split[1]], self.target]

        # Add lagged variables
        if self.variables_lags is not None: 
            df_X = add_lags(df_X, self.variables_lags)

        # Remove samples where either all features are nan or target is nan
        is_nan = df_X.isna().all(axis=1) | df_y.isna().all(axis=1)
        df_model = pd.concat([df_X, df_y], axis=1)[~is_nan]

        # Keep all timestamps for which zenith <= 100° (day timestamps)
        if self.train_on_day_only:
            idx_day = df_model[df_model['zenith'] <= 100].index
            df_model = df_model.loc[idx_day, :]

        # Create target and feature DataFrames
        if self.diff_target_with_physical:
            df_model[self.target] = df_model[self.target]-df_model['Physical_Forecast']

        # Use mean window to smooth target
        df_model[self.target] = df_model[self.target].rolling(self.target_smoothing_window, win_type='boxcar', center=True, min_periods=0).mean()

        # Apply sample weighting
        if self.weight_params:
            weight_end = self.weight_params['weight_end']
            weight_shape = self.weight_params['weight_shape']
            valid_times = df_model.index.get_level_values('valid_datetime')
            days = np.array((valid_times[-1]-valid_times).total_seconds()/(60*60*24))
            weight = (1-weight_end)*np.exp(-days/weight_shape)+weight_end
        else:
            weight = None

        return df_X, df_y, df_model, weight


    def generate_dataset_site_split(self, df, split_set='train'):
        # Generate train and valid splits

        dfs_X_site, dfs_y_site, dfs_model_site, weight_site = [], [], [], []
        print('Generating dataset...')
        with tqdm(total=len(self.sites)*len(self.splits[split_set])) as pbar:
            for site in self.sites:

                dfs_X_split, dfs_y_split, dfs_model_split, weight_split = [], [], [], []
                for split in self.splits[split_set]:
                    df_X, df_y, df_model, weight = self.generate_dataset(df, split, site)

                    dfs_X_split.append(df_X)
                    dfs_y_split.append(df_y)
                    dfs_model_split.append(df_model)
                    weight_split.append(weight)

                    pbar.update(1)

                dfs_X_site.append(dfs_X_split)
                dfs_y_site.append(dfs_y_split)
                dfs_model_site.append(dfs_model_split)
                weight_site.append(weight_split)

        return dfs_X_site, dfs_y_site, dfs_model_site, weight_site

    def build_model_dataset(self, df_model_train, df_model_valid=None, weight=None): 
        # Build up dataset adapted to models
        train_set, valid_sets = {}, {}
        if 'lightgbm' in self.model_params:
            train_set_lgb = lgb.Dataset(df_model_train[self.features], label=df_model_train[self.target], weight=weight, params={'verbose': -1}, free_raw_data=False)
            train_set['lightgbm'] = train_set_lgb
            if df_model_valid is not None: 
                valid_set_lgb = lgb.Dataset(df_model_valid[self.features], label=df_model_valid[self.target], params={'verbose': -1}, free_raw_data=False)
                valid_sets['lightgbm'] = [train_set_lgb, valid_set_lgb]
                valid_names = ['train', 'valid']
            else:
                vaild_sets['lightgbm'] = [train_set_lgb]        
                valid_names = ['train']
        if 'xgboost' in self.model_params:
            train_set_xgb = xgb.DMatrix(df_model_train[self.features], label=df_model_train[self.target], weight=weight)
            train_set['xgboost'] = train_set_xgb
            if df_model_valid is not None: 
                valid_set_xgb = xgb.DMatrix(df_model_valid[self.features], label=df_model_valid[self.target])
                valid_sets['xgboost'] = [(train_set_xgb, 'train'), (valid_set_xgb, 'valid')]
            else: 
                valid_sets['xgboost'] = [(train_set_xgb, 'train')]   
        if 'catboost' in self.model_params:
            train_set_cb = cb.Pool(df_model_train[self.features], label=df_model_train[self.target], weight=weight)
            train_set['catboost'] = train_set_cb
            if df_model_valid is not None: 
                valid_set_cb = cb.Pool(df_model_valid[self.features], label=df_model_valid[self.target])
                valid_sets['catboost'] = [train_set_cb, valid_set_cb]      
            else: 
                valid_sets['catboost'] = [train_set_cb]
        if 'skboost' in self.model_params:
            train_set['skboost'] = [df_model_train[self.features], df_model_train[self.target].squeeze(), weight]

        return train_set, valid_sets

    def train_on_objective(self, train_set, valid_sets, objective='mean', alpha=None):
        gbm, evals_result = {}, {}
        if 'lightgbm' in self.model_params:
            with warnings.catch_warnings():
                if self.model_params['lightgbm']['verbose'] == -1: 
                    warnings.simplefilter("ignore")
                if objective == 'mean': 
                    self.model_params['lightgbm']['objective'] = 'mean_squared_error'
                elif objective == 'quantile': 
                    self.model_params['lightgbm']['objective'] = 'quantile'
                    self.model_params['lightgbm']['alpha'] = alpha
                evals_result_lgb = {}
                model_lgb = lgb.train(self.model_params['lightgbm'],
                                      train_set['lightgbm'],
                                      valid_sets=valid_sets['lightgbm'],
                                      valid_names=None,
                                      evals_result=evals_result_lgb,
                                      verbose_eval=False,
                                      callbacks=None)
                gbm['lightgbm'] = model_lgb
                evals_result['lightgbm'] = evals_result_lgb
        if 'xgboost' in self.model_params:
            if objective == 'mean': 
                self.model_params['xgboost']['objective'] = 'reg:squarederror'
                evals_result_xgb = {}
                model_xgb = xgb.train(self.model_params['xgboost'],
                                      train_set['xgboost'],
                                      self.model_params['xgboost']['num_round'],
                                      evals=valid_sets['xgboost'], 
                                      evals_result=evals_result_xgb,
                                      verbose_eval=False)
                gbm['xgboost'] = model_xgb
                evals_result['xgboost'] = evals_result_xgb
        if 'catboost' in self.model_params:
            if objective == 'mean': 
                self.model_params['catboost']['objective'] = 'Lq:q=2'
            elif objective == 'quantile': 
                self.model_params['catboost']['objective'] = 'Quantile:alpha='+str(alpha)
            model_cb = cb.train(pool=train_set['catboost'],
                                params=self.model_params['catboost'],
                                eval_set=valid_sets['catboost'],
                                verbose=False)
            gbm['catboost'] = model_cb
        if 'skboost' in self.model_params:
            if objective == 'mean': 
                self.model_params['skboost']['loss'] = 'ls'
            elif objective == 'quantile': 
                self.model_params['skboost']['loss'] = 'quantile'
                self.model_params['skboost']['alpha'] = alpha
            model_skb = ensemble.GradientBoostingRegressor(**self.model_params['skboost'])
            model_skb.fit(train_set['skboost'][0], train_set['skboost'][1], sample_weight=train_set['skboost'][2])
            gbm['skboost'] = model_skb

        return gbm, evals_result

    def train(self, train_set, valid_sets): 

        gbm_q, evals_result_q = {}, {}
        if 'mean' in self.regression_params['type']:
            # Train model for mean
            gbm, evals_result = self.train_on_objective(train_set, valid_sets, objective='mean')

            gbm_q['mean'] = gbm
            evals_result_q['mean'] = evals_result

        if 'quantile' in self.regression_params['type']:
            # Train models for different quantiles
            alpha_q = np.arange(self.regression_params['alpha_range'][0],
                                self.regression_params['alpha_range'][1],
                                self.regression_params['alpha_range'][2])
            for alpha in alpha_q:
                gbm, evals_result = self.train_on_objective(train_set, valid_sets, objective='quantile', alpha=alpha)
                gbm_q['quantile'+str(alpha)] = gbm
                evals_result_q['quantile'+str(alpha)] = evals_result

        if not (('mean' in self.regression_params['type']) or ('quantile' in self.regression_params['type'])):
            raise ValueError('Value of regression parameter "objective" not recognized.')

        return gbm_q, evals_result_q

    def train_site_split(self, dfs_model_train_site, weight_train_site=None, dfs_model_valid_site=None):
        
        gbm_site, evals_result_site = [], []
        print('Training...')
        with tqdm(total=len(dfs_model_train_site)*len(dfs_model_train_site[0])) as pbar:
            for idx_site, dfs_model_train_split in enumerate(dfs_model_train_site):

                gbm_split, evals_result_split = [], []
                for idx_split, df_model_train in enumerate(dfs_model_train_split):
                    
                    if weight_train_site is not None: 
                        weight = weight_train_site[idx_site][idx_split]
                    else:
                        weight = None
                        
                    if dfs_model_valid_site is not None: 
                        df_model_valid = dfs_model_valid_site[idx_site][idx_split]
                    else:
                        df_model_valid = None

                    train_set, valid_sets = self.build_model_dataset(df_model_train, df_model_valid=df_model_valid, weight=weight)
                    gbm_q, evals_result_q = self.train(train_set, valid_sets)

                    gbm_split.append(gbm_q)
                    evals_result_split.append(evals_result_q)
                    
                    pbar.update(1)

                gbm_site.append(gbm_split)
                evals_result_site.append(evals_result_split)

        return gbm_site, evals_result_site
        

    def predict(self, df_X, gbm_q): 
        # Use trained models to predict

        def post_process(y_pred):

            if self.diff_target_with_physical: 
                y_pred = y_pred+df_X['Physical_Forecast'].values
            
            if not self.regression_params['target_min_max'] == [None, None]: 
                target_min_max = self.regression_params['target_min_max']

                if target_min_max[1] == 'clearsky': 
                    idx_clearsky = y_pred > df_X['Clearsky_Forecast'].values
                    y_pred[idx_clearsky] = df_X['Clearsky_Forecast'].values[idx_clearsky]
                    
                    if not target_min_max[0] == None:
                        y_pred = y_pred.clip(min=target_min_max[0], max=None)

                else:
                    y_pred = y_pred.clip(min=target_min_max[0], max=target_min_max[1])

            return y_pred

        # Make DataFrame to store the predictions in
        idx_q_start = 0
        columns = []
        if 'mean' in self.regression_params['type']:
            idx_q_start += 1
            columns.append('mean')

        if 'quantile' in self.regression_params['type']:
            alpha_q = np.arange(self.regression_params['alpha_range'][0],
                                self.regression_params['alpha_range'][1],
                                self.regression_params['alpha_range'][2])
            columns.extend(['quantile{0}'.format(int(round(100*alpha))) for alpha in alpha_q])
        
        df_index = pd.DataFrame(index=df_X.index, columns=columns)

        # Keep all timestamps for which zenith <= 100° (day timestamps)
        if self.train_on_day_only:
            idx_day = df_X['zenith'] <= 100
            idx_night = df_X['zenith'] > 100
            df_X = df_X[idx_day]

        y_pred_q = {}
        df_y_pred_qs = {}
        for model in self.model_params.keys():

            y_pred_q[model] = []
            for q in gbm_q.keys():
                if model == 'lightgbm':
                    y_pred = gbm_q[q][model].predict(df_X[self.features])
                elif model == 'xgboost': 
                    if self.regression_params['type'][0] == 'mean':
                        y_pred = gbm_q[q][model].predict(xgb.DMatrix(df_X[self.features]))
                elif model == 'catboost': 
                    y_pred = gbm_q[q][model].predict(df_X[self.features])
                elif model == 'skboost': 
                    y_pred = gbm_q[q][model].predict(df_X[self.features])

                y_pred = post_process(y_pred)
                y_pred_q[model].append(y_pred)

            # Convert list to numpy 2D-array
            y_pred_q[model] = np.stack(y_pred_q[model], axis=-1)

            if 'quantile_postprocess' in self.regression_params.keys():
                if self.regression_params['quantile_postprocess'] == 'none':
                    pass
                elif self.regression_params['quantile_postprocess'] == 'sorting': 
                    # Lazy post-sorting of quantiles
                    y_pred_q[model] = np.sort(y_pred_q[model], axis=-1)
                elif self.regression_params['quantile_postprocess'] == 'isotonic_regression': 
                    # Isotonic regression
                    regressor = IsotonicRegression()
                    y_pred_q[model] = np.stack([regressor.fit_transform(alpha_q, y_pred_q[model][sample,:]) for sample in range(idx_q_start, y_pred_q[model].shape[0])])                    

            # Create prediction output dataframe
            df_y_pred_q = df_index
            if self.train_on_day_only:
                df_y_pred_q[idx_day] = y_pred_q[model]
                df_y_pred_q[idx_night] = 0
            else:
                df_y_pred_q.values[:] = y_pred_q[model]

            df_y_pred_qs[model] = df_y_pred_q.astype('float64')

        df_y_pred_qs['mean'] = pd.concat([df_y_pred_qs[model] for model in df_y_pred_qs.keys()]).groupby(level=[0,1]).mean()

        return df_y_pred_qs

    def predict_site_split(self, dfs_X_site, gbm_site):
        # Use trained models to predict for their corresponding split

        dfs_y_pred_site = []
        print('Predicting...')
        with tqdm(total=len(dfs_X_site)*len(dfs_X_site[0])) as pbar:
            for dfs_X_split, gbm_split, in zip(dfs_X_site, gbm_site):

                dfs_y_pred_split = []
                for dfs_X, gbm_q in zip(dfs_X_split, gbm_split):

                    df_y_pred_q = self.predict(dfs_X, gbm_q)
                    dfs_y_pred_split.append(df_y_pred_q)

                    pbar.update(1)

                dfs_y_pred_site.append(dfs_y_pred_split)

        return dfs_y_pred_site


    def calculate_loss(self, dfs_y_true_site, dfs_y_pred_site):

        print('Calculating loss...')
        if 'mean' in self.regression_params['type']:

            dfs_loss_model = {}
            for model in self.model_params.keys():
                dfs_loss_site = []
                for dfs_y_true_split, dfs_y_pred_split in zip(dfs_y_true_site, dfs_y_pred_site):
                    dfs_loss_split = []
                    for dfs_y_true, dfs_y_pred in zip(dfs_y_true_split, dfs_y_pred_split):
                        y_true = dfs_y_true[self.target].values
                        df_pred = dfs_y_pred[model]
                        y_pred = dfs_y_pred[model].values

                        loss = (df_y_pred-df_y_true)**2

                        df_loss = pd.DataFrame(data=loss, index=df_pred.index, columns=df_pred.columns)
                        
                        dfs_loss_split.append(df_loss)

                    dfs_loss_site.append(dfs_loss_split)

                dfs_loss_model[model] = dfs_loss_site

        if 'quantile' in self.regression_params['type']:
            # Evaluation using pinball loss function

            alpha_q = np.arange(self.regression_params['alpha_range'][0],
                                self.regression_params['alpha_range'][1],
                                self.regression_params['alpha_range'][2])
            a = alpha_q.reshape(1,-1)

            dfs_loss_model = {}
            for model in self.model_params.keys():

                dfs_loss_site = []
                for dfs_y_true_split, dfs_y_pred_split in zip(dfs_y_true_site, dfs_y_pred_site):
                    dfs_loss_split = []
                    for df_y_true, df_y_pred in zip(dfs_y_true_split, dfs_y_pred_split):
                        y_true = df_y_true[self.target].values
                        df_pred = df_y_pred[model]
                        y_pred = df_pred.values

                        # Pinball loss with nan if true label is nan
                        with np.errstate(invalid='ignore'):
                            loss = np.where(np.isnan(y_true),
                                            np.nan,
                                            np.where(y_true < y_pred,
                                                    (1-a)*(y_pred-y_true),
                                                    a*(y_true-y_pred)))

                            df_loss = pd.DataFrame(data=loss, index=df_pred.index, columns=df_pred.columns)

                        dfs_loss_split.append(df_loss)

                    dfs_loss_site.append(dfs_loss_split)

                dfs_loss_model[model] = dfs_loss_site
        
        return dfs_loss_model


    def calculate_score(self, dfs_loss_model):

        flatten = lambda l: [item for sublist in l for item in sublist]
        score_model = {}
        for model in self.model_params.keys():
            score_model[model] = pd.concat(flatten(dfs_loss_model[model])).mean().mean()

        return score_model


    def save_result(self, params_json, result_data, result_prediction, result_model, result_evals, result_loss):

        print('Saving results...')
        trial_path = self.path_result+self.trial_name
        if os.path.exists(trial_path):
            shutil.rmtree(trial_path)
        os.makedirs(trial_path)

        file_name_json = '/params_'+self.trial_name+'.json'
        with open(trial_path+file_name_json, 'w') as file:
            json.dump(params_json, file, indent=4)

        result = {}
        if self.save_options['data'] == True:
            for key in result_data.keys():
                os.makedirs(trial_path+'/'+key)
                for site in range(len(result_data[key])):
                    for split in range(len(result_data[key][0])):
                        file_name_result = key+'_site_{0}_split_{1}.csv'.format(site, split)
                        result_data[key][site][split].to_csv(trial_path+'/'+key+'/'+file_name_result)
        if self.save_options['prediction'] == True:
            for key in result_prediction.keys():
                os.makedirs(trial_path+'/'+key)
                for site in range(len(result_prediction[key])):
                    for split in range(len(result_prediction[key][0])):
                        for model in self.model_params.keys():
                            file_name_result = key+'_'+model+'_site_{0}_split_{1}.csv'.format(site, split)
                            result_prediction[key][site][split][model].to_csv(trial_path+'/'+key+'/'+file_name_result)
        if self.save_options['model'] == True:
            for key in result_model.keys():
                os.makedirs(trial_path+'/'+key)
                for site in range(len(result_model[key])):
                    for split in range(len(result_model[key][0])):
                        for q in result_model[key][0][0].keys():
                            for model in self.model_params.keys():
                                if model in ['lightgbm', 'xgboost', 'catboost']: 
                                    file_name_result = key+'_'+model+'_'+str(q)+'_site_{0}_split_{1}.txt'.format(site, split)
                                    result_model[key][site][split][q][model].save_model(trial_path+'/'+key+'/'+file_name_result)
                                if model == 'skboost': 
                                    file_name_result = key+'_'+model+'_q_'+str(q)+'_site_{0}_split_{1}.pkl'.format(site, split)
                                    with open(trial_path+'/'+key+'/'+file_name_result, 'wb') as f:
                                        pickle.dump(result_model[key][site][split][q][model], f)
        if self.save_options['loss'] == True:
            for key in result_loss.keys():
                os.makedirs(trial_path+'/'+key)
                for model in self.model_params.keys():
                    # Need to flip list to concatenate on sites
                    #TODO change order of list when creating them instead
                    dfs_loss_site = result_loss[key][model]
                    dfs_loss_split = list(map(list, zip(*dfs_loss_site)))
                    for split in range(len(dfs_loss_split)):
                        file_name_loss = key+'_'+model+'_split_{0}.csv'.format(split)
                        df_loss = pd.concat(dfs_loss_split[split], axis=1, keys=self.sites)
                        df_loss.to_csv(trial_path+'/'+key+'/'+file_name_loss, header=True)
        if self.save_options['overall_score'] == True:
            score_train_model = self.calculate_score(result_loss['dfs_loss_train_site'])
            score_valid_model = self.calculate_score(result_loss['dfs_loss_valid_site'])
            file_name_score = self.path_result+'/trial-scores.txt'

            for model in score_train_model.keys():
                if not os.path.exists(file_name_score):
                    with open(file_name_score, 'w') as file:
                        file.write('Name: {0}; Comment: {1}; Model: {2}; Train score {3}; valid score {4};\n'.format(self.trial_name, self.trial_comment, model, score_train_model[model], score_valid_model[model]))
                else:
                    with open(file_name_score, 'a') as file:
                        file.write('Name: {0}; Comment: {1}; Model: {2}; Train score {3}; valid score {4};\n'.format(self.trial_name, self.trial_comment, model, score_train_model[model], score_valid_model[model]))

        print('Results saved to: '+trial_path)


def main(df, params_json):
    trial = Trial(params_json)

    dfs_X_train_site, dfs_y_train_site, dfs_model_train_site, weight_train_site = trial.generate_dataset_site_split(df, split_set='train')
    dfs_X_valid_site, dfs_y_valid_site, dfs_model_valid_site, weight_valid_site = trial.generate_dataset_site_split(df, split_set='valid')
    
    gbm_site, evals_result_site = trial.train_site_split(dfs_model_train_site, weight_train_site=weight_train_site, dfs_model_valid_site=dfs_model_valid_site)
    
    dfs_y_pred_train_site = trial.predict_site_split(dfs_X_train_site, gbm_site)
    dfs_y_pred_valid_site = trial.predict_site_split(dfs_X_valid_site, gbm_site)
    
    dfs_loss_train_site = trial.calculate_loss(dfs_y_train_site, dfs_y_pred_train_site)
    dfs_loss_valid_site = trial.calculate_loss(dfs_y_valid_site, dfs_y_pred_valid_site)

    result_data = {'dfs_X_train_site': dfs_X_train_site,
                   'dfs_X_valid_site': dfs_X_valid_site,
                   'dfs_y_train_site': dfs_y_train_site,
                   'dfs_y_valid_site': dfs_y_valid_site}
    result_prediction = {'dfs_y_pred_train_site': dfs_y_pred_train_site,
                         'dfs_y_pred_valid_site': dfs_y_pred_valid_site}
    result_model = {'gbms': gbm_site}
    result_evals = {'evals_result_site': evals_result_site}
    result_loss = {'dfs_loss_train_site': dfs_loss_train_site,
                   'dfs_loss_valid_site': dfs_loss_valid_site}

    trial.save_result(params_json, result_data, result_prediction, result_model, result_evals, result_loss)

if __name__ == '__main__':
    params_path = sys.argv[1]
    with open(params_path, 'r', encoding='utf-8') as file:
        params_json = json.loads(file.read())

    df = load_data(params_json['path_preprocessed_data']+params_json['filename_preprocessed_data'])
    main(df, params_json)
