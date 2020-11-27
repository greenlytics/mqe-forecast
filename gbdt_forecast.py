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
import joblib 

from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import catboost as cb
from sklearn import ensemble
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb

class Trial():
    def __init__(self, params_json):
        self.params_json = params_json

        # Mandatory input variables
        self.trial_name = params_json['trial_name']
        self.trial_comment = params_json['trial_comment']
        self.path_result = params_json['path_result']
        self.trial_path = self.path_result+self.trial_name
        self.path_preprocessed_data = params_json['path_preprocessed_data']
        self.filename_preprocessed_data = params_json['filename_preprocessed_data']
        self.sites = params_json['sites']
        self.features = params_json['features']
        self.target = params_json['target']
        self.model_params = params_json['model_params']
        self.regression_params = params_json['regression_params']
        self.save_options = params_json['save_options']
        
        if 'parallel_processing' in params_json:
            self.parallel_processing = params_json['parallel_processing']
        else:
            self.parallel_processing = {'backend': 'threading',
                                        'n_workers': 1}
            
        if 'quantile' in self.regression_params['type']:
            self.alpha_q = np.arange(self.regression_params['alpha_range'][0],
                                     self.regression_params['alpha_range'][1],
                                     self.regression_params['alpha_range'][2])
            if len(self.alpha_q) == 0: 
                raise ValueError('Number of quantiles needs to be larger than zero.')

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
        if 'train_only_zenith_angle_below' in params_json:
            self.train_only_zenith_angle_below = params_json['train_only_zenith_angle_below']
        else: 
            self.train_only_zenith_angle_below = False
        if 'time_weight_params' in params_json: 
            self.time_weight_params = params_json['time_weight_params']
        else:
            self.time_weight_params = False
        if 'target_level_weight_params' in params_json: 
            self.target_level_weight_params = params_json['target_level_weight_params']
        else:
            self.target_level_weight_params = False     
        if 'custom_weight_params' in params_json: 
            self.custom_weight_params = params_json['custom_weight_params']
        else:
            self.custom_weight_params = False   

        if 'datetime_splits' in params_json: 
            self.datetime_splits = params_json['datetime_splits']
            self.splits = params_json['datetime_splits']

        elif 'train_test_splits' in params_json:
            pass
        elif 'crossvalidation_splits' in params_json:
            pass
        else:
            raise ValueError('One of `datetime_splits`, `train_test_splits` or `crossvalidation_splits` must be given in params_json.')

        # Runtime
        self.parallel_backend = params_json.get("parallel_backend", "threading")


    def load_data(self):
        # Load preprocessed data
        path_data = self.path_preprocessed_data+self.filename_preprocessed_data
        df = pd.read_csv(path_data, header=[0,1], index_col=[0,1], parse_dates=True)

        return df


    def generate_dataset(self, df, split, site): 

        def add_lags(df, variables_lags): 
            # Lagged features
            vspec = pd.DataFrame([(k, lag) for k, v in variables_lags.items() for lag in v], columns=["Variable", "Lag"]) \
                                 .set_index("Variable") \
                                 .sort_values("Lag")
            for lag, variables in vspec.groupby("Lag").groups.items():
                shifted = df.loc[:, sorted(variables)].groupby('ref_datetime').shift(lag)
                shifted.columns = ['%s_lag%s' % (variable, lag) for variable in sorted(variables)]
                df = pd.concat([df, shifted], axis=1)

            return df

        # Make target into list if not already
        if self.diff_target_with_physical and not ('Physical_Forecast' in self.features):
            df_X = df[site].loc[pd.IndexSlice[:, split[0]:split[1]], self.features+['Physical_Forecast']]
        else:
            df_X = df[site].loc[pd.IndexSlice[:, split[0]:split[1]], self.features]

        df_y = df[site].loc[pd.IndexSlice[:, split[0]:split[1]], [self.target]]

        # Add lagged variables
        if self.variables_lags is not None: 
            df_X = add_lags(df_X, self.variables_lags)

        # Remove samples where either all features are nan or target is nan
        is_nan = df_X.isna().all(axis=1) | df_y.isna().all(axis=1)
        df_model = pd.concat([df_X, df_y], axis=1)[~is_nan]

        # Keep all timestamps for which zenith <= prescribed value (day timestamps)
        if self.train_only_zenith_angle_below:
            idx_day = df_model[df_model['zenith'] <= self.train_only_zenith_angle_below].index
            df_model = df_model.loc[idx_day, :]

        # Create target and feature DataFrames
        if self.diff_target_with_physical:
            df_model[self.target] = df_model[self.target]-df_model['Physical_Forecast']

        # Use mean window to smooth target
        df_model[self.target] = df_model[self.target].rolling(self.target_smoothing_window, win_type='boxcar', center=True, min_periods=0).mean()

        # Apply time-based sample weighting
        if self.time_weight_params:
            weight_end = self.time_weight_params['weight_end']
            weight_shape = self.time_weight_params['weight_shape']
            valid_times = df_model.index.get_level_values('valid_datetime')
            days = np.array((valid_times[-1]-valid_times).total_seconds()/(60*60*24))
            time_weight = (1-weight_end)*np.exp(-days/weight_shape)+weight_end
        else:
            time_weight = 1

        # Apply target level-based sample weighting
        if self.target_level_weight_params:
            weight_end = self.target_level_weight_params['weight_end']
            weight_shape = self.target_level_weight_params['weight_shape']
            target = df_model[self.target]
            target_min = target.min()
            target_max = target.max()
            b = (1-weight_end)/(np.exp(-target_min/weight_shape)-np.exp(-target_max/weight_shape))
            a = weight_end+b*np.exp(-target_min/weight_shape)
            level_weight = a-b*np.exp(-target/weight_shape)
        else: 
            level_weight = 1

        # Apply custom sample weighting
        if self.custom_weight_params:
            df_custom_weight = df[self.custom_weight_params['column']]
            custom_weight = df_custom_weight[df_model.index].values
        else:
            custom_weight = 1

        weight = time_weight*level_weight*custom_weight

        return df_X, df_y, df_model, weight


    def generate_dataset_split_site(self, df, split_set='train'):
        # Generate train and valid splits

        print('Generating dataset...')
        dfs_X_split, dfs_y_split, dfs_model_split, weight_split = [], [], [], []
        with tqdm(total=len(self.splits[split_set])*len(self.sites)) as pbar:
            for split in self.splits[split_set]:
                dfs_X_site, dfs_y_site, dfs_model_site, weight_site = [], [], [], []
                for site in self.sites:

                    df_X, df_y, df_model, weight = self.generate_dataset(df, split, site)

                    dfs_X_site.append(df_X)
                    dfs_y_site.append(df_y)
                    dfs_model_site.append(df_model)
                    weight_site.append(weight)

                    pbar.update(1)

                dfs_X_split.append(dfs_X_site)
                dfs_y_split.append(dfs_y_site)
                dfs_model_split.append(dfs_model_site)
                weight_split.append(weight_site)

        return dfs_X_split, dfs_y_split, dfs_model_split, weight_split

    def build_model_dataset(self, df_model_train, model, df_model_valid=None, weight=None): 
        # Build up dataset adapted to models
        train_set, valid_sets = {}, {}
        if model == 'lightgbm':
            train_set = lgb.Dataset(df_model_train[self.features], label=df_model_train[[self.target]], weight=weight, params={'verbose': -1}, free_raw_data=False)
            if df_model_valid is not None: 
                valid_set = lgb.Dataset(df_model_valid[self.features], label=df_model_valid[[self.target]], params={'verbose': -1}, free_raw_data=False)
                valid_sets = [train_set, valid_set]
            else:
                vaild_sets['lightgbm'] = [train_set_lgb]        
        elif model == 'xgboost':
            train_set = xgb.DMatrix(df_model_train[self.features], label=df_model_train[[self.target]], weight=weight)
            if df_model_valid is not None: 
                valid_set = xgb.DMatrix(df_model_valid[self.features], label=df_model_valid[[self.target]])
                valid_sets = [(train_set, 'train'), (valid_set, 'valid')]
            else: 
                valid_sets = [(train_set, 'train')]   
        elif model == 'catboost':
            train_set = cb.Pool(df_model_train[self.features], label=df_model_train[[self.target]], weight=weight)
            if df_model_valid is not None: 
                valid_set = cb.Pool(df_model_valid[self.features], label=df_model_valid[[self.target]])
                valid_sets = [valid_set]      
        elif model == 'skboost' in self.model_params:
            train_set = [df_model_train[self.features], df_model_train[self.target], weight]

        return train_set, valid_sets

    def train_on_objective(self, df_model_train, model, objective='mean', alpha=None, df_model_valid=None, weight=None):

        if model == 'lightgbm':
            if objective == 'mean': 
                objective_lgb = 'mean_squared_error'
                eval_key_name = 'l2'
            elif objective == 'quantile': 
                objective_lgb = 'quantile'
                eval_key_name = 'quantile'
                self.model_params['lightgbm']['alpha'] = alpha
            else: 
                raise ValueError("'objective' for lightgbm must be either 'mean' or 'quantile'")
            self.model_params['lightgbm']['objective'] = objective_lgb
                            
            gbm = lgb.LGBMRegressor(**self.model_params['lightgbm'])
            gbm.fit(df_model_train[self.features],
                    df_model_train[[self.target]],
                    sample_weight=weight,
                    eval_set=[(df_model_valid[self.features], df_model_valid[[self.target]])],
                    eval_names=None,
                    eval_metric=objective_lgb,
                    early_stopping_rounds=self.model_params['lightgbm'].get("early_stopping", None),
                    verbose=False,
                    categorical_feature='', #TODO Change to pass this parameter explicitly
                    callbacks=None)

            evals_result = {key: value[eval_key_name] for key, value in gbm.evals_result_.items()}

        elif model == 'xgboost':
            if objective == 'mean': 
                self.model_params['xgboost']['objective'] = 'reg:squarederror'
            else: 
                raise ValueError("'objective' for xgboost must be 'mean'.")
            evals_result = {}
            gbm = xgb.train(self.model_params['xgboost'],
                            train_set,
                            self.model_params['xgboost']['num_round'],
                            evals=valid_sets, 
                            evals_result=evals_result,
                            verbose_eval=False)
            evals_result = None #TODO Add evals for xgboost

        elif model=='catboost':
            if objective == 'mean': 
                objective_cb = 'Lq:q=2'
            elif objective == 'quantile': 
                objective_cb = 'Quantile:alpha={0:g}'.format(alpha)
            else: 
                raise ValueError("'objective' must be one of ['mean', 'quantile']")
            self.model_params['catboost']['objective'] = objective_cb

            gbm = cb.train(pool=train_set,
                           params=self.model_params['catboost'],
                           eval_set=valid_sets,
                           verbose=False)
            evals_result = {key: value[objective_cb] for key, value in gbm.evals_result_.items()}

        elif model=='skboost':
            if objective == 'mean': 
                self.model_params['skboost']['loss'] = 'ls'
                self.model_params['skboost']['criterion'] = 'friedman_mse'
            elif objective == 'quantile': 
                self.model_params['skboost']['loss'] = 'quantile'
                self.model_params['skboost']['alpha'] = alpha
                self.model_params['skboost']['criterion'] = 'mae' #TODO Check how `criterion` affects quantile loss.
            else: 
                raise ValueError("'objective' must be one of ['mean', 'quantile']")

            gbm = ensemble.GradientBoostingRegressor(**self.model_params['skboost'])
            gbm.fit(train_set[0], train_set[1], sample_weight=train_set[2])
            evals_result = None #TODO Add evals for skboost

        else: 
            raise ValueError("'objective' for skboost must be either 'mean' or 'quantile'")
        
        return gbm, evals_result

    def train(self, df_model_train, model, df_model_valid=None, weight=None): 

        gbm_q, evals_result_q = {}, {}
        if 'mean' in self.regression_params['type']:
            # Train model for mean
            gbm, evals_result = self.train_on_objective(df_model_train, model, objective='mean', df_model_valid=df_model_valid, weight=weight)
            gbm_q['mean'] = gbm
            evals_result_q['mean'] = evals_result #TODO change this similarly to quantile case. 

        if 'quantile' in self.regression_params['type']:
            # Train models for different quantiles
            with joblib.parallel_backend(self.parallel_processing['backend']):
                results = joblib.Parallel(n_jobs=self.parallel_processing['n_workers'])(
                            joblib.delayed(self.train_on_objective)(df_model_train,
                                                                        model,
                                                                        objective='quantile',
                                                                        alpha=alpha,
                                                                        df_model_valid=df_model_valid, 
                                                                        weight=weight)
                            for alpha in self.alpha_q)

            for (gbm, evals_result), alpha in zip(results, self.alpha_q):
                gbm_q['quantile{0:.2f}'.format(alpha)] = gbm
                evals_result_q['quantile{0:.2f}'.format(alpha)] = evals_result

        if not (('mean' in self.regression_params['type']) or ('quantile' in self.regression_params['type'])):
            raise ValueError('Value of regression parameter "objective" not recognized.')

        return gbm_q, evals_result_q

    def train_model_split_site(self, dfs_model_train_split, dfs_model_valid_split=None, weight_train_split=None):
        
        print('Training...')
        gbm_model, evals_result_model = {}, {}
        with tqdm(total=len(self.model_params.keys())*len(dfs_model_train_split)*len(dfs_model_train_split[0])) as pbar:
            for model in self.model_params.keys():
                gbm_split, evals_result_split = [], []
                for idx_split, dfs_model_train_site in enumerate(dfs_model_train_split):

                    gbm_site, evals_result_site = [], []
                    for idx_site, df_model_train in enumerate(dfs_model_train_site):
                            
                        if dfs_model_valid_split is not None: 
                            df_model_valid = dfs_model_valid_split[idx_split][idx_site]
                        else:
                            df_model_valid = None

                        if weight_train_split is not None: 
                            weight = weight_train_split[idx_split][idx_site]
                        else:
                            weight = None
                        
                        #train_set, valid_sets = self.build_model_dataset(df_model_train, model, df_model_valid=df_model_valid, weight=weight)
                        gbm_q, evals_result_q = self.train(df_model_train, model, df_model_valid=df_model_valid, weight=weight) #TODO Make it possible to train starting from an existing model. E.g. LightGBM has a `input_model` option. 

                        #TODO Add support for categorical_features. 
                        gbm_site.append(gbm_q)
                        evals_result_site.append(evals_result_q)
                        
                        pbar.update(1)

                    gbm_split.append(gbm_site)
                    evals_result_split.append(evals_result_site)
                
                gbm_model[model] = gbm_split
                evals_result_model[model] = evals_result_split

        return gbm_model, evals_result_model
        

    def predict(self, df_X, gbm_q, model): 
        # Use trained models to predict
        #TODO Use SHAP to estimate contribution of different features. https://github.com/slundberg/shap

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
            columns.extend(['quantile{0}'.format(int(round(100*alpha))) for alpha in self.alpha_q])
        
        df_index = pd.DataFrame(index=df_X.index, columns=columns)

        # Keep all timestamps for which zenith <= prescribed value (day timestamps)
        if self.train_only_zenith_angle_below:
            idx_day = df_X['zenith'] <= self.train_only_zenith_angle_below
            idx_night = df_X['zenith'] > self.train_only_zenith_angle_below
            df_X = df_X[idx_day]

        df_y_pred_qs = {}

        y_pred_q = []
        for q in gbm_q.keys():
            if model == 'lightgbm':
                y_pred = gbm_q[q].predict(df_X[self.features])
            elif model == 'xgboost': 
                if self.regression_params['type'][0] == 'mean':
                    y_pred = gbm_q[q].predict(xgb.DMatrix(df_X[self.features]))
            elif model == 'catboost': 
                y_pred = gbm_q[q].predict(df_X[self.features])
            elif model == 'skboost': 
                y_pred = gbm_q[q].predict(df_X[self.features])
            else:
                raise ValueError()

            y_pred = post_process(y_pred)
            y_pred_q.append(y_pred)

        # Convert list to numpy 2D-array
        y_pred_q = np.stack(y_pred_q, axis=-1)

        if 'quantile_postprocess' in self.regression_params.keys():
            if self.regression_params['quantile_postprocess'] == 'none':
                pass
            elif self.regression_params['quantile_postprocess'] == 'sorting': 
                # Lazy post-sorting of quantiles
                y_pred_q = np.sort(y_pred_q, axis=-1)
            elif self.regression_params['quantile_postprocess'] == 'isotonic_regression': 
                # Isotonic regression
                regressor = IsotonicRegression()
                y_pred_q = np.stack([regressor.fit_transform(self.alpha_q, y_pred_q[sample,:]) for sample in range(idx_q_start, y_pred_q.shape[0])])                    

        # Create prediction output dataframe
        df_y_pred_q = df_index
        if self.train_only_zenith_angle_below:
            df_y_pred_q[idx_day] = y_pred_q
            df_y_pred_q[idx_night] = 0
        else:
            df_y_pred_q.values[:] = y_pred_q

        df_y_pred_q = df_y_pred_q.astype('float64')

        return df_y_pred_q

    def predict_model_split_site(self, dfs_X_split, gbm_model):
        # Use trained models to predict for their corresponding split

        dfs_y_pred_model = {}
        print('Predicting...')
        with tqdm(total=len(self.model_params.keys())*len(dfs_X_split[0])*len(dfs_X_split)) as pbar:
            for model in self.model_params.keys():
                dfs_y_pred_split = []
                gbm_split = gbm_model[model]
                for dfs_X_site, gbm_site in zip(dfs_X_split, gbm_split):
                    dfs_y_pred_site = []
                    for dfs_X, gbm_q, in zip(dfs_X_site, gbm_site):
                        df_y_pred_q = self.predict(dfs_X, gbm_q, model)
                        dfs_y_pred_site.append(df_y_pred_q)

                        pbar.update(1)

                    dfs_y_pred_split.append(dfs_y_pred_site)
                
                dfs_y_pred_model[model] = dfs_y_pred_split

        return dfs_y_pred_model


    def calculate_loss(self, df_y_true, df_y_pred): 

        if 'mean' in self.regression_params['type']:
            y_true = df_y_true[[self.target]].values
            y_pred = df_y_pred['mean'].values
            loss = (y_pred-y_true)**2
            df_loss_mean = pd.DataFrame(data=loss, index=df_y_pred.index, columns=['mean'])
        else:
            df_loss_mean = None

        if 'quantile' in self.regression_params['type']:
            a = self.alpha_q.reshape(1,-1)
            y_true = df_y_true[[self.target]].values
            y_pred = df_y_pred.filter(regex='quantile').values

            # Pinball loss with nan if true label is nan
            with np.errstate(invalid='ignore'):
                loss = np.where(np.isnan(y_true),
                                np.nan,
                                np.where(y_true < y_pred,
                                        (1-a)*(y_pred-y_true),
                                        a*(y_true-y_pred)))

                df_loss_quantile = pd.DataFrame(data=loss, index=df_y_pred.index, columns=df_y_pred.filter(regex='quantile').columns)
        else:
            df_loss_quantile = None
        
        df_loss = pd.concat([df_loss_mean, df_loss_quantile], axis=1)
    
        return df_loss

    def calculate_loss_split_site(self, dfs_y_true_split, dfs_y_pred_model):

        print('Calculating loss...')

        dfs_loss_model = {}
        for model in self.model_params.keys():
            dfs_loss_split = []
            dfs_y_pred_split = dfs_y_pred_model[model]
            for dfs_y_true_site, dfs_y_pred_site in zip(dfs_y_true_split, dfs_y_pred_split):
                dfs_loss_site = []
                for df_y_true, df_y_pred in zip(dfs_y_true_site, dfs_y_pred_site):

                    df_loss = self.calculate_loss(df_y_true, df_y_pred)
                    
                    dfs_loss_site.append(df_loss)

                dfs_loss_split.append(dfs_loss_site)

            dfs_loss_model[model] = dfs_loss_split
        
        return dfs_loss_model


    def calculate_score(self, dfs_loss_model):

        flatten = lambda l: [item for sublist in l for item in sublist]
        score_model = {}
        for model in self.model_params.keys():
            score_model[model] = pd.concat(flatten(dfs_loss_model[model])).mean().mean()

        return score_model

    def save_json(self):
        if os.path.exists(self.trial_path):
            shutil.rmtree(self.trial_path)
        os.makedirs(self.trial_path)

        file_name_json = '/params_'+self.trial_name+'.json'
        with open(self.trial_path+file_name_json, 'w') as file:
            json.dump(params_json, file, indent=4)

    def save_data_prediction_evals_loss(self, df, key, model, split, site): 
        file_name = key+'_'+model+'_split_{0}_site_{1}.csv'.format(split, site)
        df.to_csv(self.trial_path+'/'+key+'/'+file_name)

    def save_model(self, gbm_q, key, model, split, site):
        for q in gbm_q.keys():
            gbm = gbm_q[q]
            if model in ['lightgbm', 'xgboost', 'catboost']: 
                file_name = key+'_'+model+'_q_'+q+'_split_{0}_site_{1}.txt'.format(split, site)
                gbm.booster_.save_model(self.trial_path+'/'+key+'/'+file_name)
            if model == 'skboost': 
                file_name = key+'_'+model+'_q_'+q+'_split_{0}_site_{1}.pkl'.format(split, site)
                with open(self.trial_path+'/'+key+'/'+file_name, 'wb') as f:
                    pickle.dump(gbm, f)

    def save_result(self, params_json, result_data, result_prediction, result_model, result_evals, result_loss):

        print('Saving results...')
        self.save_json()

        if self.save_options['data'] == True:
            for key in result_data.keys():
                os.makedirs(self.trial_path+'/'+key)
                for split in range(len(result_data[key])):
                    df = pd.concat(result_data[key][split], axis=1, keys=self.sites)
                    self.save_data_prediction_evals_loss(df, key, 'none', split, 'all') 
 
        if self.save_options['prediction'] == True:
            for key in result_prediction.keys():
                os.makedirs(self.trial_path+'/'+key)
                for model in self.model_params.keys():
                    for split in range(len(result_prediction[key][model])):
                        df = pd.concat(result_prediction[key][model][split], axis=1, keys=self.sites)
                        self.save_data_prediction_evals_loss(df, key, model, split, 'all')      

        if self.save_options['model'] == True:
            for key in result_model.keys():
                os.makedirs(self.trial_path+'/'+key)
                for model in self.model_params.keys():
                    for split in range(len(result_model[key][model])):
                        for site in range(len(result_model[key][model][0])):
                            gbm_q = result_model[key][model][split][site]
                            self.save_model(gbm_q, key, model, split, site)

        if self.save_options['evals'] == True:
            for key in result_evals.keys():
                os.makedirs(self.trial_path+'/'+key)
                for model in self.model_params.keys():
                    for split in range(len(result_evals[key][model])):
                        data = result_evals[key][model][split]
                        data = {(level1_key, level2_key, level3_key): pd.Series(values)
                                for level1_key, level2_dict in zip(self.sites,data)
                                for level2_key, level3_dict in level2_dict.items()
                                for level3_key, values in level3_dict.items()}
                        df = pd.DataFrame(data)
                        df.index.name = 'trees'
                        self.save_data_prediction_evals_loss(df, key, model, split, 'all')      

        if self.save_options['loss'] == True:
            for key in result_loss.keys():
                os.makedirs(self.trial_path+'/'+key)
                for model in self.model_params.keys():
                    for split in range(len(result_loss[key][model])):
                        df = pd.concat(result_loss[key][model][split], axis=1, keys=self.sites)
                        self.save_data_prediction_evals_loss(df, key, model, split, 'all')      

        if self.save_options['overall_score'] == True:
            score_train_model = self.calculate_score(result_loss['dfs_loss_train'])
            score_valid_model = self.calculate_score(result_loss['dfs_loss_valid'])
            file_name = self.path_result+'/trial-scores.txt'

            for model in score_train_model.keys():
                if not os.path.exists(file_name):
                    with open(file_name, 'w') as file:
                        file.write('Name: {0}; Comment: {1}; Model: {2}; Train score {3}; valid score {4};\n'.format(self.trial_name, self.trial_comment, model, score_train_model[model], score_valid_model[model]))
                else:
                    with open(file_name, 'a') as file:
                        file.write('Name: {0}; Comment: {1}; Model: {2}; Train score {3}; valid score {4};\n'.format(self.trial_name, self.trial_comment, model, score_train_model[model], score_valid_model[model]))
        else:
            score_train_model = None
            score_valid_model = None
        print('Results saved to: '+self.trial_path)

        return score_train_model, score_valid_model

    def run(self, df):

        print('Running trial pipeline for trial: {0}...'.format(self.trial_name))
        dfs_X_train_split, dfs_y_train_split, dfs_model_train_split, weight_train_split = self.generate_dataset_split_site(df, split_set='train')
        dfs_X_valid_split, dfs_y_valid_split, dfs_model_valid_split, _ = self.generate_dataset_split_site(df, split_set='valid')

        gbm_model, evals_result_model = self.train_model_split_site(dfs_model_train_split, dfs_model_valid_split=dfs_model_valid_split, weight_train_split=weight_train_split)

        dfs_y_pred_train_model = self.predict_model_split_site(dfs_X_train_split, gbm_model)
        dfs_y_pred_valid_model = self.predict_model_split_site(dfs_X_valid_split, gbm_model)

        dfs_loss_train_model = self.calculate_loss_split_site(dfs_y_train_split, dfs_y_pred_train_model)
        dfs_loss_valid_model = self.calculate_loss_split_site(dfs_y_valid_split, dfs_y_pred_valid_model)

        result_data = {'dfs_X_train': dfs_X_train_split,
                       'dfs_X_valid': dfs_X_valid_split,
                       'dfs_y_train': dfs_y_train_split,
                       'dfs_y_valid': dfs_y_valid_split}
        result_model = {'gbm_model': gbm_model}
        result_evals = {'evals_result': evals_result_model}
        result_prediction = {'dfs_y_pred_train': dfs_y_pred_train_model,
                             'dfs_y_pred_valid': dfs_y_pred_valid_model}
        result_loss = {'dfs_loss_train': dfs_loss_train_model,
                       'dfs_loss_valid': dfs_loss_valid_model}

        score_train_model, score_valid_model = self.save_result(self.params_json, result_data, result_prediction, result_model, result_evals, result_loss)

        return score_train_model, score_valid_model

    def run_parallel(self, df):
        print('Running trial pipeline for trial: {0}...'.format(self.trial_name))

        self.save_json()
        
        if self.save_options['model'] == True:
            os.makedirs(self.trial_path+'/'+'gbm_model')
        if self.save_options['prediction'] == True:
            os.makedirs(self.trial_path+'/'+'df_y_pred_train')
            os.makedirs(self.trial_path+'/'+'df_y_pred_valid')
        if self.save_options['loss'] == True:
            os.makedirs(self.trial_path+'/'+'df_loss_train')
            os.makedirs(self.trial_path+'/'+'df_loss_valid')
        if self.save_options['data'] == True:
            os.makedirs(self.trial_path+'/'+'df_X_train')
            os.makedirs(self.trial_path+'/'+'df_X_valid')
            os.makedirs(self.trial_path+'/'+'df_y_train')
            os.makedirs(self.trial_path+'/'+'df_y_valid')

  
        def train_site(df, split_idx, split_train, split_valid, site, pbar):
            df_X_train, df_y_train, df_model_train, weight = self.generate_dataset(df, split_train, site)
            df_X_valid, df_y_valid, df_model_valid, _ = self.generate_dataset(df, split_valid, site)

            for model in self.model_params.keys():

                gbm_q, evals_result_q = self.train(df_model_train, model, df_model_valid=df_model_valid, weight=weight)

                df_y_pred_train = self.predict(df_X_train, gbm_q, model)
                df_y_pred_valid = self.predict(df_X_valid, gbm_q, model)

                df_loss_train = self.calculate_loss(df_y_train, df_y_pred_train)
                df_loss_valid = self.calculate_loss(df_y_valid, df_y_pred_valid)

                if self.save_options['model'] == True:
                    self.save_model(gbm_q, 'gbm_model', model, split_idx, site)
                if self.save_options['prediction'] == True:
                    self.save_data_prediction_evals_loss(df_y_pred_train, 'df_y_pred_train', model, split_idx, site)
                    self.save_data_prediction_evals_loss(df_y_pred_valid, 'df_y_pred_valid', model, split_idx, site)
                if self.save_options['evals'] == True:
                    self.save_data_prediction_evals_loss(evals_result_q, key, model, split, 'all')
                if self.save_options['loss'] == True:
                    self.save_data_prediction_evals_loss(df_loss_train, 'df_loss_train', model, split_idx, site)
                    self.save_data_prediction_evals_loss(df_loss_valid, 'df_loss_valid', model, split_idx, site)

                pbar.update(1)

            if self.save_options['data'] == True:
                self.save_data_prediction_evals_loss(df_X_train, 'df_X_train', 'none', split_idx, site)      
                self.save_data_prediction_evals_loss(df_X_valid, 'df_X_valid', 'none', split_idx, site)      
                self.save_data_prediction_evals_loss(df_y_train, 'df_y_train', 'none', split_idx, site)      
                self.save_data_prediction_evals_loss(df_y_valid, 'df_y_valid', 'none', split_idx, site) 

            return True 

        with tqdm(total=len(self.splits['train'])*len(self.sites)) as pbar:
            with joblib.parallel_backend(self.parallel_processing['backend']):
                results = joblib.Parallel(n_jobs=self.parallel_processing['n_workers'])(
                            joblib.delayed(train_site)(df, split_idx, split_train, split_valid, site, pbar)
                            for split_idx, (split_train, split_valid) in enumerate(zip(self.splits['train'], self.splits['valid']))
                                for site in self.sites)

        print("Results: ", results)
                    
                           

if __name__ == '__main__':
    params_path = sys.argv[1]
    with open(params_path, 'r', encoding='utf-8') as file:
        params_json = json.loads(file.read())

    trial = Trial(params_json)
    df = trial.load_data()
    trial.run_parallel(df)
