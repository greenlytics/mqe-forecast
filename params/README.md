# Parameters
This page contains descriptions of all parameters in `gbdt-forecast` as well as useful links and general advice on tuning GBDT models. 

## `gbdt-forecast` parameters
* `trial_name`: The name of the trial. Will be used as file name to store the result.
* `trial_comment`: Comment about the trial setup.
* `path_result`: Path to where the result will be stored.
* `path_raw_data`: Path to the raw input data.
* `path_preprocessed_data`: Path to the preprocessed input data.
* `site_coords`: List of lists of coordinates pairs of the sites in the form `[[longitude_1, latitude_1], [longitude_2, latitude_2], ...]`.
* `site_altitude`: List with altitude of the sites in meter. Used for physical solar power forecasting. 
* `site_capacity`: List with farm capacities in same unit as original power time series. Used for normalisation. 
* `panel_orientation`: List with orientation of the panels (calculated clockwise and North is 0°). Used for physical solar power forecasting. 
* `panel_tilt`: List with tilt of the panels (0° is horisontal). Used for physical solar power forecasting.
* `splits`: A dictionary with where keys can be `train`, `valid`, `test`. Each value is a list of lists of splits (start and end time) on the form `[[start_time_1, end_time_1], [start_time_2, end_time_2], ...]`. Strings `start_time` and `end_time` should have the format `YYYY-mm-dd HH:MM:SS` and is assumed to be UTC. 
* `sites`: List of name of the sites to train on. Sites names corresponding to names of columns in preprocessed data.
* `features`: List of features to use for model prediction. Feature names corresponding to names of columns in preprocessed data.
* `variables_lags`: Dictionary of feature-lags pairs `[{feature_1: lags_1}, {feature_2: lags_2}, ...]` where `feature` is a feature from the `features` list and `lags` is a list of lags (non-zero, positive or negative integers) to include as additional `features` in the model.  
* `target`: String with name of the power target variable. Target name corresponding to name of column in preprocessed data.
* `diff_target_with_physical`: Boolean (`false` or `true`) if to use physical model as base model and learn the residuals with gradient boosting decision tree model.
* `target_smoothing_window`: Default 1. Window to smooth the target variable before training. Smoothing is done with a centered boxcar window. Should be an odd number for window to be centered.
* `train_on_day_only`: Boolean (`false` or `true`) if to only train on daytime data for which `zentih < 100°`.
* `regression_params`:

  * `type`: Type of regression. Either `mean` or `quantile`.
  * `alpha_range`: Range of quantiles on the form `[start, stop, step]` creates a list of quantiles through `numpy.arange(start, stop, step)`.
  * `y_min_max`: List with min and max values [`y_min`, `y_max`] to clip model predictions. If `Clearsky_Forecast` is in `features` then it can be used as upper limit by setting `y_max="clearsky"`. Set to `[null, null]` to disable clipping of predictions.

* `model_params`: Model parameters given to LightGBM. See LightGBM documentation.
* `weight_params`: Allows to weight recent samples in training data more compared to outdated samples. Applies an exponential decay on the form `weight = (1-weight_end)*numpy.exp(-days/weight_shape)+weight_end`, where `days` are number of days from the most recent sample.

  * `weight_end`: Weight of the most outdated sample. Should be a number in the range [0,1]. Set to 1 to disable sample weighting.
  * `weight_shape`: Shape of the exponential weighting function.

* `save_result`: Dictionary with keys according to below.

  * `prediction`: Boolean if to save predictions to result.
  * `model`: Boolean if to save models to result.
  * `loss`: Boolean if to save loss to result.
  * `overall_score`: Boolean if to save overall score to result.

## Advice on tuning GBDT models
Quick start receipt for training GBDT models:
##### 1) Set number of trees to something as high as possible (e.g. 3000)
##### 2) Run a grid search or random search
##### 3) Finally set number of trees even higher and tune learning rate

The maximum depth of the tree controls the degree of feature interaction that you can model. Usually it is fair to assume that the degree if interactions is fairly low. As a rule of thumb the depth of the tree should be around 4-6 [1]

Shrinkage (or learning rate) is a parameter that exponentially reduced the weight that a tree will have in the final prediction as more and more trees are added. As a general rule of thumb, a model with higher shrinkage (or low learning rate) and more trees will perform better than a model with low shrinkage and few trees. The learning rate should typically be less than 0.1.

Stochastic gradient boosting is doing the same thing as random forest is doing. Either sampling data points or sampling feature set before creating a split point. This typically leads to improved accuracy.

## Useful links
Learning parameters for the supported gradient boosting implementations can be found here:

#### [LightGBM](https://github.com/microsoft/LightGBM)

* [Parameter documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
* [Laurae++ Interactive Documentation](https://sites.google.com/view/lauraepp/parameters)

More links: 
[LightGBM, Official documentation on parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
[LightGBM, Parameters tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
[What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
[Lauraepp, Parameters](https://sites.google.com/view/lauraepp/parameters?authuser=0)
<br> Comparison between XGBoost and LightGBM parameters.

#### [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html)

* [Parameter documentation](https://xgboost.readthedocs.io/en/latest/parameter.html)
* [Laurae++ Interactive Documentation](https://sites.google.com/view/lauraepp/parameters)