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

# get hyperparameters


# evaluate
lr_model_perf, lr_strat_perf = evaluate_models(lr_models, X_train, y_train, X_test, y_test, strategy_train, strategy_test)

# insert model type into results for concatenation later
lr_model_perf.insert(0, 'model', 'LogReg')
lr_strat_perf.insert(0, 'model', 'LogReg')


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

# evaluate
rf_model_perf, rf_strat_perf = evaluate_models(rf_models, X_train, y_train, X_test, y_test, strategy_train, strategy_test)
