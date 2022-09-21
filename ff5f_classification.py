##### Import Libraries #####

# Data Manipulation Libraries
import pandas as pd 
import numpy as np
from numpy.random import seed # for reproducible results
seed(1)
import warnings
warnings.filterwarnings("ignore")

# Data visualization
import matplotlib.pyplot as plt
%matplotlib inline 

# Machine Learning libraries
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV, PredefinedSplit
import tensorflow as tf
tf.random.set_seed(1) # for reproducible results
from tensorflow import keras

# System libraries
import os # for tensorboard to visualize training
from joblib import dump, load
from math import floor, ceil

# Stats libraries
from scipy.stats import uniform, randint, t
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Custom helper functions
from ff5f_classification_helper_funcs import *

##### Load Data #####

X = pd.read_csv('features.csv').set_index("yyyymm")
y = pd.read_csv('targets.csv').set_index("yyyymm")


##### Prepare Data #####

# Split
# note: strategy_train/test is the same a y_train/test but with returns rather than directions
# for evaluating strategies
X_train, X_val, X_test, y_train, y_val, y_test, strategy_train, strategy_test = fractional_train_val_test_split(X, y)

# standardize
X_train, X_val, X_test = standardize_data(X_train, X_val, X_test)

# Create full training sets and create predefined split (necessary for sklearn's hyperparam optimization funcs)
X_train_full = X_train.append(X_val)
y_train_full = y_train.append(y_val)
pds=create_predefined_validation_split(X_train, X_val)


##### Models #####

# Evaluate naive buy-and-hold strategy as baseline
naive_model_perf, naive_strat_perf, naive_strategies, is_naive_strategies = evaluate_naive_strategy(y_test, strategy_test, strategy_train)


## Logistic Regression w/ elastic net regularization

# grids for hyperparam optimization
lr_param_dist = {'l1_ratio': uniform(loc=0.0, scale=1.0),
                           'C' : uniform(loc=0.0, scale=3.0)}
lr_grid = {
    'l1_ratio' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'C' : [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
}

# initialize estimator
lr_en_model = LogisticRegression(penalty='elasticnet', solver='saga', warm_start=True)

# fit models
lr_models = fit_models(lr_en_model, lr_grid, X_train_full, y_train_full, pds, searcher='grid')
# note that models are refit on the full training set with the "optimal" hyperparameters

# get hyperparameters
lr_params = get_sklearn_model_params(lr_models)

# evaluate models
lr_model_perf, lr_strat_perf = evaluate_models(lr_models, X_train, y_train, X_test, y_test, strategy_train, strategy_test)

# inserts a column specifying the model type, e.g. LogReg. 
# this is done so that all model performance can be combined for comparison and sorting.
lr_model_perf.insert(1, 'model', 'LogReg')
lr_strat_perf.insert(1, 'model', 'LogReg')

lr_perf = pd.merge(lr_model_perf, lr_strat_perf, on=['target', 'model'])

# save models
save_models(lr_models, 'lr_')

# write results to excel
write_to_excel(pd.DataFrame(lr_params), sheet_name='LogReg Params')
write_to_excel(lr_perf, sheet_name='LogReg Performance')



## Random Forest

# param grid
rf_param_dist = {
    'n_estimators' : [100, 200, 500, 1000, 2000, 5000, 10000],
    'max_depth' : list(range(1, 10+1))
}

# init estimator
rf = RandomForestClassifier(random_state=1)

# fit
rf_models = fit_models(rf, rf_param_dist, X_train_full, y_train_full, pds, searcher='grid', verbose=True)

# get hyperparams
rf_params = get_sklearn_model_params(rf_models)

# evaluate
rf_model_perf, rf_strat_perf = evaluate_models(rf_models, X_train, y_train, X_test, y_test, strategy_train, strategy_test)

rf_model_perf.insert(1, 'model', 'RF')
rf_strat_perf.insert(1, 'model', 'RF')

rf_perf = pd.merge(rf_model_perf, rf_strat_perf, on=['target', 'model'])

# write results to excel
write_to_excel(pd.DataFrame(rf_params), sheet_name='RF Params')
write_to_excel(rf_perf, sheet_name='RF Performance')



## Gradient Boosted Trees

gbt_param_dist = {
    'learning_rate' : uniform(0.0, 0.1),
    'n_estimators' : [20000],
    'subsample' : uniform(0.1, 0.5),
    'max_depth' : [1, 2],
    'validation_fraction' : uniform(0.1, 0.5)
}
gbt_param_grid = {
    'learning_rate' : [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'n_estimators' : [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    'subsample' : [0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth' : [1, 2, 3, 4, 5]
}

# init
gbt = GradientBoostingClassifier(random_state=1)

# fit
gbt_models = fit_models(gbt, gbt_param_grid, X_train_full, y_train_full, pds, searcher='grid', verbose=True)

# get hparams
gbt_params = get_sklearn_model_params(gbt_models)

# evaluate
gbt_model_perf, gbt_strat_perf = evaluate_models(gbt_models, X_train, y_train, X_test, y_test, strategy_train, strategy_test)

gbt_model_perf.insert(1, 'model', 'GBT')
gbt_strat_perf.insert(1, 'model', 'GBT')

gbt_perf = pd.merge(gbt_model_perf, gbt_strat_perf, on=['target', 'model'])

# save models
save_models(gbt_models, 'gbt_')

# write results to excel
write_to_excel(pd.DataFrame(gbt_params), sheet_name='GBT Params')
write_to_excel(gbt_perf, sheet_name='GBT Performance')



## Extremely Randomized Trees

extra_tree_param_grid = {
    'n_estimators' : [1000, 2000, 5000, 10000, 20000],
    'max_depth' : [1, 2, 3, 4, 5]
}

# initialize and fit
extra_tree = ExtraTreesClassifier(random_state=1)

extra_trees = fit_models(extra_tree, extra_tree_param_grid, X_train_full, y_train_full, pds, searcher='grid', verbose=True)

# get hparams
xt_params = get_sklearn_model_params(extra_trees)

# evaluate 
xt_model_perf, xt_strat_perf = evaluate_models(extra_trees, X_train, y_train, X_test, y_test, strategy_train, strategy_test)

xt_model_perf.insert(1, 'model', 'XT')
xt_strat_perf.insert(1, 'model', 'XT')

xt_perf = pd.merge(xt_model_perf, xt_strat_perf, on=['target', 'model'])

# save and write

save_models(xt_models, 'xt_')

write_to_excel(pd.DataFrame(xt_params), sheet_name='XT Params')
write_to_excel(xt_perf, sheet_name='XT Performance')



## Neural Networks

# callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=30, restore_best_weights=True)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=15)

# Constant nets (i.e., 1 layer, 32 nodes)

layers_to_try = [1, 2, 3, 4, 5]
nodes_to_try = [4, 8, 16, 32, 64, 128, 256]

fit_all_kinds_of_nets(layers=layers_to_try, nodes=nodes_to_try, node_config='constant')

# Pyrimidical nets with 256 nodes in the first layer
fit_all_kinds_of_nets(layers=layers_to_try, nodes=nodes_to_try, node_config='halving', first_layer_nodes=256) 

# Pyrimidical nets with 128 nodes in the first layer
fit_all_kinds_of_nets(layers=layers_to_try, nodes=nodes_to_try, node_config='halving', first_layer_nodes=128) 

# Pyrimidical nets with 64 nodes in the first layer
fit_all_kinds_of_nets(layers=layers_to_try, nodes=nodes_to_try, node_config='halving', first_layer_nodes=64) 

# Pyrimidical nets with 32 nodes in the first layer
fit_all_kinds_of_nets(layers=layers_to_try, nodes=nodes_to_try, node_config='halving', first_layer_nodes=256) 

