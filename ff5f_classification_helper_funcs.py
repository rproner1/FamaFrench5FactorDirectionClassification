# Data Manipulation Libraries
import pandas as pd 
import numpy as np
from numpy.random import seed # for reproducible results
seed(1)
import warnings
warnings.filterwarnings("ignore")

# Data visualization
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score, r2_score
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




def standardize_data(X_train, X_val, X_test):
    
    """ 
    standardizes each column of the data to have a mean 0 and a standard deviation of 1 
    
    Parameters
    ----------
    X_train: the feature training set.
    X_val: the feature validation set.
    X_test: the feature testing set.

    Returns
    ----------
    X_train: the feature training set (scaled).
    X_val: the feature validation set (scaled).
    X_test: the feature testing set (scaled).

    """
    
    mean = X_train.mean()
    std = X_train.std()
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_val, X_test

def custom_train_val_test_split(X, y, n_val_years=10, train_cutoff=198701):
    
    """ 
    Creates a custom (temporal) train/val/test split 
    
    Parameters
    ----------
    X: the feature set.
    y: the target set.
    n_val_years: the number of years to use for the validation set (default=10).
    train_cutoff: a date to cutoff the training set at in yyyymm int format (default=198701).

    Returns
    ----------
    X_train: the training set of features.
    X_val: the validation set of features.
    X_test: the testing set of features.
    y_train: the training set of targets.
    y_val: the validation set of targets.
    y_test: the testing set of targets.
    strategy_train: the training set of targets but with returns rather than directions.
    strategy_test: the testing set of targets but with returns rather than directions.
    """
    
    val_cutoff = train_cutoff + n_val_years*100
    
    X_train = X.loc[ : train_cutoff ]
    X_val = X.loc[ (train_cutoff + 1) : val_cutoff]
    X_test = X.loc[ (val_cutoff + 1) : ]
    
    # We want to predict one month ahead so we need to add one to the cutoffs
    train_cutoff += 1
    val_cutoff += 1
    
    strategy_test = y.loc[ (val_cutoff + 1) : ]
    strategy_train = y.loc[ : val_cutoff ]
    
    y_01 = y.copy()
    y_01[y_01 >= 0] = 1 # up or no change
    y_01[y_01 < 0] = 0 # down
    
    y_train = y_01.loc[ : train_cutoff ]
    y_val = y_01.loc[ (train_cutoff + 1) : val_cutoff] 
    y_test = y_01.loc[ (val_cutoff + 1) : ]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, strategy_train, strategy_test

def fractional_train_val_test_split(X, y, validation_split=0.15, full_train_cutoff=200001):
    
    """ 
    Creates a custom (temporal) train/val/test split based on proportion of the full training set
    to be used for validation.
    
    Parameters
    ----------
    X: the feature set.
    y: the target set.
    validation_split: the fraction of the total training set to use for the validation set (default=0.15).
    full_train_cutoff: a date to cutoff the full training set at in yyyymm int format (default=200001).

    Returns
    ----------
    X_train: the training set of features.
    X_val: the validation set of features.
    X_test: the testing set of features.
    y_train: the training set of targets.
    y_val: the validation set of targets.
    y_test: the testing set of targets.
    strategy_train: the training set of targets but with returns rather than directions.
    strategy_test: the testing set of targets but with returns rather than directions.
    """
    
    X_train_full = X.loc[ : full_train_cutoff ]
    
    indexable_rows = len(X_train_full)-1 # rows we can index by position, e.g. index 0, 1, ..., len(X_train_full) - 1
    train_cutoff = floor( indexable_rows*(1-validation_split) ) # multiply the indexable rows by 1-validation_split giving the number of (indexable) rows in train
    
    X_train = X_train_full.iloc[ : train_cutoff ] # further divide full train into a training set containing the first 85% of rows 
    X_val = X_train_full.iloc[  train_cutoff : ] # val set containing last 15% of rows. Note that slicing with iloc does not include the upper bound so we don't add 1 as with loc
    X_test = X.loc[ (full_train_cutoff+1) : ] # test set with the rest of the data.
    
    # We want to predict one month ahead so we need to add one to the cutoff in order for y to have the same number of rows
    str_date = str(full_train_cutoff)
    month = int(str_date[-2:])
    if month == 12:
        full_train_cutoff = full_train_cutoff + 100 - 11 # Add 1 year, subtract 11 months to arrive at January the next year
    else:
        full_train_cutoff += 1
    
    strategy_train = y.iloc[ : train_cutoff ]
    strategy_test = y.loc[ (full_train_cutoff + 1) : ]
    
    y_01 = y.copy()
    y_01[y_01 >= 0] = 1 # up or no change
    y_01[y_01 < 0] = 0 # down
    
    y_train_full = y_01.loc[ : full_train_cutoff ]
    
    y_train = y_train_full.iloc[ : train_cutoff ]
    y_val = y_train_full.iloc[ train_cutoff :  ]
    y_test = y_01.loc[ (full_train_cutoff + 1) : ]
    
    if len(X_train_full) != len(y_train_full):
        print('X_train_full and y_train_full have unequal lengths.')
    
    if len(X_train) != len(y_train):
        print("X_train and y_train lengths are not equal!")
    
    if len(X_val) != len(y_val):
        print('X_val and y_val have unequal lengths.')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, strategy_train, strategy_test


def create_multilabel_target(y):
    
    """ 
    
    Creates a multilabel target matrix for multiclass classification. 
    
    Parameters
    ----------
    y: the target data set.

    Returns
    ----------
    y_multilabel: a multilabel np.array for multilabel classification.     
    """
    
    y_multilabel = np.c_[y['mktrf'], y['smb'], y['hml'], y['rmw'], y['cma'], y['umd']]
    
    return y_multilabel

def create_predefined_validation_split(X_train, X_val):
    
    """ 
    Creates predefined validation split for sklearn hyperparameter search classes. 
    
    Parameters
    ----------
    X_train: the training set of features.
    X_val: the validation set of features.

    Returns
    ----------
    predefined_split: a PredefinedSplit object.   

    """
    
    X_train_full = X_train.append(X_val)
    
    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if x in X_train.index else 0 for x in X_train_full.index]

    # Use the list to create PredefinedSplit
    predefined_split = PredefinedSplit(test_fold = split_index)
    
    return predefined_split

def fit_models(estimator, param_dist, X_train, y_train, predefined_split, searcher='grid', verbose=False):
    
    """ 

    Fits a model for each variable using Successive Halving Random Search for each variable in y_train. 
    
    Parameters
    ----------
    estimator: a sklearn estimator object.
    param_dist: a dictionary of hyperparameter names as keys and distributions (for halving random search)
        or lists of values (for grid search) as values. 
    X_train: training set of features.
    y_train: training set of target variables.
    predefined_split: a PredefinedSplit for model validation.
    searcher: the hyperparameter optimization method to use {'grid', 'halving'} (default='grid').
    verbose: prints the target variable being fit for.
    
    returns
    ----------
    best_models: a dictionary, indexable by target variable, of the optimal models.

    """



    best_models = {}

    if searcher == 'grid':
        for col in y_train.columns:
            if verbose:
                print(f'Now fitting model for {col}.')
            model_search = GridSearchCV(estimator, param_dist, cv=predefined_split, refit=True)
            model_fit = model_search.fit(X_train, y_train[col])
            best_models[col] = model_fit
    elif searcher == 'halving':
        for col in y_train.columns:
            if verbose:
                print(f'Now fitting model for {col}.')
            model_search = HalvingRandomSearchCV(estimator, param_dist, n_candidates='exhaust', factor=2, resource='n_samples',  cv=predefined_split, refit=True)
            model_fit = model_search.fit(X_train, y_train[col])
            best_models[col] = model_fit
        
    return best_models


def evaluate_naive_strategy(y_test, y_train, strategy_test, strategy_train):
    
    """
    
    Evaluates the naive (buy and hold) strategy.
    
    Parameters 
    ----------
    y_test: a target df, with columns containing target variables.
    strategy_train: the training set of targets but with returns rather than directions.
    strategy_test: the testing set of targets but with returns rather than directions.

    Returns
    ----------
    model_perf_df: a dataframe with model performance metrics.
    strat_perf_df: a dataframe with strategy performance metrics.
    naive_strategies: oos naive strategies
    is_naive_strategies: is naive strategies
    
    """
    
    model_performance = {
        'model' : list(y_test.columns),
        'is_acc' : [],
        'is_prec' : [],
        'is_roc_auc' : [],
        'oos_acc' : [],
        'oos_prec' : [],
        'oos_roc_auc' : []
    }
    strat_performance = {
        'model' : list(y_test.columns),
        'is_mean' : [],
        'is_std' : [],
        'is_sharpe' : [],
        'oos_mean' : [],
        'oos_std' : [],
        'oos_sharpe' : []
    }
    
    naive_strategies = {}
    is_naive_strategies = {}
    
    for col in y_test.columns:
    
        naive_preds = np.ones(len(strategy_test))
        naive_acc = accuracy_score(y_test[col], naive_preds)
        naive_prec = precision_score(y_test[col], naive_preds)
        naive_roc_auc = roc_auc_score(y_test[col], naive_preds)

        naive_strategy = strategy_test[col] * naive_preds
        naive_mean_ret = naive_strategy.mean()*12 # annualized mean return
        naive_std = naive_strategy.std()*np.sqrt(12) # annualized standard deviation
        naive_sharpe = naive_mean_ret / naive_std # annualized sharpe ratio
        
        is_naive_preds = np.ones(len(strategy_train))
        is_naive_acc = accuracy_score(y_train[col], is_naive_preds)
        is_naive_prec = precision_score(y_train[col], is_naive_preds)
        is_naive_roc_auc = roc_auc_score(y_train[col], is_naive_preds)
        
        is_naive_strategy = strategy_train[col] * is_naive_preds
        is_naive_mean = is_naive_strategy.mean()*12 # annualized in-sample mean
        is_naive_std = is_naive_strategy.std()*np.sqrt(12) # annualized in-sample std
        is_naive_sharpe = is_naive_mean / is_naive_std # in-sample sharpe
        
        # record out-of-sample results
        model_performance['oos_acc'].append(naive_acc), model_performance['oos_prec'].append(naive_prec), model_performance['oos_roc_auc'].append(naive_roc_auc)
        strat_performance['oos_mean'].append(naive_mean_ret), strat_performance['oos_std'].append(naive_std), strat_performance['oos_sharpe'].append(naive_sharpe)
        
        # record in-sample results
        model_performance['is_acc'].append(is_naive_acc), model_performance['is_prec'].append(is_naive_prec), model_performance['is_roc_auc'].append(is_naive_roc_auc)
        strat_performance['is_mean'].append(is_naive_mean), strat_performance['is_std'].append(is_naive_std), strat_performance['is_sharpe'].append(is_naive_sharpe)
        
        # record naive_strategies
        naive_strategies[col] = naive_strategy
        is_naive_strategies[col] = is_naive_strategy
        
    model_perf_df = pd.DataFrame(model_performance)
    strat_perf_df = pd.DataFrame(strat_performance)
    
    return model_perf_df, strat_perf_df, naive_strategies, is_naive_strategies


def evaluate_models(models, X_train, y_train, X_test, y_test, strategy_train, strategy_test):
    
    """ 
    
    Evaluates a set of models in-sample and out-of-sample. 
    
    Parameters
    ----------
    models: a dict of models indexable by target variable.
    X_train: a training set of features.
    y_train: a training set of targets. 
    X_test: a testing set of features. 
    y_test: a testing set of targets.  
    strategy_train: the training set of targets but with returns rather than directions.
    strategy_test: the testing set of targets but with returns rather than directions.

    Returns
    ----------
    model_perf_df: Performance of the machine learning models.
    strategy_perf_df: Performance of the strategy generated by the models

    """
    
    naive_model_perf_df, naive_strat_perf_df, naive_strategies, is_naive_strategies = evaluate_naive_strategy(y_test, y_train, strategy_test, strategy_train)
    
    model_performance = {
        'target' : list(models.keys()),
        'is_acc' : [],
        'is_prec' : [],
        'is_roc_auc' : [],
        'oos_acc' : [],
        'oos_prec' : [],
        'oos_roc_auc' : []
    }
    strat_performance = {
        'target' : list(models.keys()),
        'is_mean' : [],
        'is_std' : [],
        'is_sharpe' : [],
        'is_alpha' : [],
        'is_alpha_t' : [],
        'is_beta' : [],
        'is_beta_t' : [],
        'is_r2' : [],
        'oos_mean' : [],
        'oos_std' : [],
        'oos_sharpe' : [],
        'oos_alpha' : [],
        'oos_alpha_t' : [],
        'oos_beta' : [],
        'oos_beta_t' : [],
        'oos_r2' : []
    }
    
    feature_importances = {}
    
    for var in models.keys():
        # Select model
        model = models[var]
        
        # Model metrics
        is_preds = model.predict(X_train)
        if is_preds.ndim == 2: # Allows for evaluating predictions of neural networks which return predictions as 2D arrays (len(X_test),1)
            is_preds = is_preds[:,0] # Extract the first and only column to obtain an (len(X_test),) array (len(X_test) element vector)
        oos_preds = model.predict(X_test)
        if oos_preds.ndim == 2: 
            oos_preds = oos_preds[:,0]
            
        # compute in-sample and out-of-sample model metrics
        oos_acc, is_acc = accuracy_score(y_test[var], oos_preds), accuracy_score(y_train[var], is_preds) # in/out-of-sample accuracy
        oos_prec, is_prec = precision_score(y_test[var], oos_preds), precision_score(y_train[var], is_preds) # in/out-of-sample precision 
        oos_roc_auc, is_roc_auc = roc_auc_score(y_test[var], oos_preds), roc_auc_score(y_train[var], is_preds) # in/out-of-sample ROC AUC
        
        model_performance['is_acc'].append(is_acc)
        model_performance['is_prec'].append(is_prec)
        model_performance['is_roc_auc'].append(is_roc_auc) 
        model_performance['oos_acc'].append(oos_acc)
        model_performance['oos_prec'].append(oos_prec)
        model_performance['oos_roc_auc'].append(oos_roc_auc) 
        
        # compute in-sample and out-of-sample trading strategy profitability
        oos_strategy, is_strategy = strategy_test[var] * oos_preds , strategy_train[var] * is_preds # Generate in-sample and out-of-sample strategies
        oos_mean_ret, is_mean_ret = np.mean(oos_strategy)*12, np.mean(is_strategy)*12 # annualized return of strategy
        oos_std_ret, is_std_ret = np.std(oos_strategy)*np.sqrt(12), np.std(is_strategy)*np.sqrt(12) # annualized standard deviation of strategy
        oos_sharpe, is_sharpe = oos_mean_ret / oos_std_ret, is_mean_ret / is_std_ret # sharpe ratio
        
        naive_strategy = naive_strategies[var].values.reshape(-1,1)
        is_naive_strategy = is_naive_strategies[var].values.reshape(-1,1)
        
        oos_lr = OLS(oos_strategy, add_constant(naive_strategy)).fit()
        oos_params = oos_lr.params
        

        oos_alpha = oos_params[0]
        oos_beta = oos_params[1]
        t_vals = oos_lr.tvalues
        oos_alpha_t = t_vals[0]
        oos_beta_t = t_vals[1]
        oos_r2 = oos_lr.rsquared
        
        is_lr = OLS(is_strategy, add_constant(is_naive_strategy)).fit()
        is_params = is_lr.params
        is_alpha = is_params[0]
        is_beta = is_params[1]
        t_vals = is_lr.tvalues
        is_alpha_t = t_vals[0]
        is_beta_t = t_vals[1]
        is_r2 = is_lr.rsquared
        
        
        strat_performance['is_mean'].append(is_mean_ret) 
        strat_performance['is_std'].append(is_std_ret) 
        strat_performance['is_sharpe'].append(is_sharpe)
        strat_performance['is_alpha'].append(is_alpha)
        strat_performance['is_alpha_t'].append(is_alpha_t)
        strat_performance['is_beta'].append(is_beta)
        strat_performance['is_beta_t'].append(is_beta_t)
        strat_performance['is_r2'].append(is_r2)
        strat_performance['oos_mean'].append(oos_mean_ret)
        strat_performance['oos_std'].append(oos_std_ret)
        strat_performance['oos_sharpe'].append(oos_sharpe)
        strat_performance['oos_alpha'].append(oos_alpha)
        strat_performance['oos_alpha_t'].append(oos_alpha_t)
        strat_performance['oos_beta'].append(oos_beta)
        strat_performance['oos_beta_t'].append(oos_beta_t)
        strat_performance['oos_r2'].append(oos_r2)
        
    model_perf_df = pd.DataFrame(model_performance)
    strategy_perf_df = pd.DataFrame(strat_performance)
        
    return model_perf_df, strategy_perf_df


def get_model_params(models):
    
    """ 
    
    Returns model parameters for each model in models. 
    
    Parameters 
    ----------
    models: a dict of fitted models.
    
    Returns
    ----------
    None
    
    """
    
    for model in models:
        print(models[model].best_params_)
    
    return None

def save_models(models, file_prefix):
    
    """ 
    
    Saves models. 
    
    Parameters 
    ----------
    models: a dict of fitted models.
    file_prefix: the prefix for all the saved files.
    
    Returns
    ----------
    None
    
    """
    
    for model in models:
        dump(models[model], file_prefix + model + '.joblib')
    
    return None

def expanding_window_fc(X, y, estimator, var, init_full_train_cutoff=200001, validation_split=0.15, verbose=True):
    
    """ 
    
    Performs recursive out-of-sample forecasting 
    
    Parameters 
    ----------
    X: Full feature set.
    y: Full target set.
    estimator: A sklearn estimator.
    var: the name of the target variable {'mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd'}
    init_full_train_cutoff: initial cutoff date for the full training set. Should match full_train_cutoff
        in fractional_train_val_test_split.
    validation_split: the percentage of observations of the total training set to be used for validation.
    verbose: Prints the year that is being predicted for.
    
    Returns
    ----------
    y_pred: the predictions generated by the model.
    
    Notes:
    enabling warm_start should make this faster.
    
    """
    # Initial prediction for init_full_train_cutoff + 1
    init_pred_date = init_full_train_cutoff + 1
    init_X_test = X.loc[ init_pred_date ].values.reshape(1, -1) #.values is necessary because .reshape is not applicable to series. When
    # we index a single sample we get a series
    
    y_pred = estimator.predict(init_X_test)
    
    full_train_cutoff = init_full_train_cutoff + 1
    
    while full_train_cutoff < 202112:
        
        if verbose:
            print(f'Now forecasting for {full_train_cutoff+1}')
    
        X_train, X_val, X_test, y_train, y_val, y_test, strategy_train, strategy_test = proportional_train_val_test_split(X, y, validation_split, full_train_cutoff)
    
        X_train_full, y_train_full = X_train.append(X_val), y_train.append(y_val)
    
        estimator.fit(X_train_full, y_train_full[var])
       
        str_date = str(full_train_cutoff)
        month = int(str_date[-2:])
        if month == 12:
            pred_date = full_train_cutoff + 100 - 11 # Add 1 year, subtract 11 months to arrive at January the next year
        else:
            pred_date = full_train_cutoff + 1
        
        X_test = X_test.loc[ pred_date ].values.reshape(1,-1)
        
        pred = estimator.predict(X_test)
        
        y_pred = np.append(y_pred, pred)
        
        full_train_cutoff = pred_date
    
    return y_pred

def generate_expanding_window_fcs(models, X, y, init_full_train_cutoff=200001, validation_split=0.15, verbose=True):
    
    """ 
    
    Performs recursive out-of-sample forecasting 
    
    Parameters 
    ----------
    models: a dict of (prefit on <= init_full_train_cuoff) models indexable by target.
    X: Full feature set.
    y: Full target set.
    init_full_train_cutoff: initial cutoff date for the full training set. Should match full_train_cutoff
        in fractional_train_val_test_split.
    validation_split: the percentage of observations of the total training set to be used for validation.
    verbose: Prints the year that is being predicted for.
    
    Returns
    ----------
    fcs: a dict of predictions for each model indexable by target variable.
    
    Notes:
    enabling warm_start should make this faster.
    
    """

    fcs={}
    
    for var in y.columns:
        model = models[var]
        fc = expanding_window_fc(X, y, model, var)
        fcs[var] = fc
    
    return fcs

def evaluate_expanding_window_fc(y_test, strategy_test, strategy_train, y_pred_dict):
    
    """ 
    
    Evaluates predictions generated using an expanding window.  
    
    Parameters 
    ----------
    y_test: target testing set.
    strategy_train: the training set of targets but with returns rather than directions.
    strategy_test: the testing set of targets but with returns rather than directions.
    y_pred_dict: a dict of predictions, indexable by target.
    
    Returns
    ----------
    model_met_df: Performance of the machine learning models.
    strat_perf_df: Performance of the strategy generated by the models
    
    """


    naive_model_perf_df, naive_strat_perf_df, naive_strategies, is_naive_strategies = evaluate_naive_strategy(y_test, strategy_test, strategy_train)
    
    model_metrics = {
        'oos_acc' : [],
        'oos_prec' : [],
        'oos_roc_auc' : []
    }
    strategy_perf = {
        'oos_mean' : [],
        'oos_std' : [],
        'oos_sharpe' : [],
        'oos_alpha' : [],
        'oos_beta' : [],
        'oos_r2' : []
    }
    for key in y_pred_dict:
        y_true = y_test[key]
        strat_test = strategy_test[key]
        y_pred = y_pred_dict[key]
        
        model_metrics['oos_acc'].append(accuracy_score(y_true, y_pred))
        model_metrics['oos_prec'].append(precision_score(y_true, y_pred))
        model_metrics['oos_roc_auc'].append(roc_auc_score(y_true, y_pred))
    
        strategy = strat_test * y_pred
        mean = np.mean(strategy) * 12
        std = np.std(strategy) * np.sqrt(12)
        strategy_perf['oos_mean'].append(mean) # annualized mean return
        strategy_perf['oos_std'].append(std) # annualized std
        strategy_perf['oos_sharpe'].append(mean / std) # oos sharpe ratio
        
    
    model_met_df = pd.DataFrame(model_metrics)
    strat_perf_df = pd.DataFrame(strategy_perf)
    
    return model_met_df, strat_perf_df
    
def compute_95CI(mean, sd, n, alpha):
    pass

def permutation_importance(models, X_train, y_train, X_test, y_test, strategy_train, strategy_test, n_iter=100):
    
    
    """
    Computes feature importances for a set of models.
    
    Parameters
    ----------
    models: a dict of models, with target variable names as keys.
    X_train: the training feature set. Only passed to use the evaluate_models function.
    y_train: the training target set. Only passed to use the evaluate_models function.
    X_test: the testing feature set.
    y_test: the testing target set.
    strategy_train: the training set of returns used to assess strategy performance. Only passed to use the evaluate_models function.
    strategy_test: the testing set of returns used to assess strategy performnace. Only passed to use the evaluate_models function.
    n_iter: the number of times to permute a column and predict and evaluate the predictions (default=100).

    Returns
    ----------
    feature_importances:  dict of arrays of feature importances indexable by target variable.
    
    """
    
    # Initialize feature importances dictionary.
    # Each model will have a dictionary of mean accuracy and sharpe decrease and their standard deviations
    feature_importances = {}
    
    # Compute the model performance and strategy performance for each target variable for a baseline.
    actual_model_perf, actual_strat_perf = evaluate_models(models, X_train, y_train, X_test, y_test, strategy_train, strategy_test)
    
    # Set the target variable as the index
    actual_model_perf = actual_model_perf.set_index('target')
    actual_strat_perf = actual_strat_perf.set_index('target')
    
    acc_feat_importances = {}
    sharpe_feat_importances = {}
    
    # for each model, for each feature, randomly permute that feature and re-valuate the model on the test set with the permuted feature
    for key in models:
        model = models[key]
        
        # Put the singluar model in a dictionary so it is iterable and can be passed to the evaluate models function
        model_dict = {}
        model_dict[key] = model
        
        # Extract the actual accuracy and sharpe ratio
        actual_acc = actual_model_perf.loc[key, 'oos_acc']
        actual_sharpe = actual_strat_perf.loc[key, 'oos_sharpe']
        
        # Initialize lists which will contain each variables mean and std dec in accuracy and sharpe ratio 
        model_acc_differences = []
        model_acc_diff_sd = []
        model_sharpe_differences = []
        model_sharpe_diff_sd = []
        
        # Initialize dictionary to store the above data
        model_feat_import_dict = {}
        
        for var in X_test.columns:
            
            var_differences_in_acc = []
            var_differences_in_sharpe = []
            
            # repeat permutations n_iter times
            for i in range(n_iter):
                # permute the values of the feature
                X_test[var] = np.random.permutation(X_test[var].values)

                # predict on the new test features
                preds = model.predict(X_test)

                # evaluate predictions
                perm_model_perf, perm_strat_perf = evaluate_models(model_dict, X_train, y_train, X_test, y_test, strategy_train, strategy_test)
                perm_model_perf = perm_model_perf.set_index('target')
                perm_strat_perf = perm_strat_perf.set_index('target')

                # Get the accuracy and sharpe ratio
                perm_acc = perm_model_perf.loc[key, 'oos_acc']
                perm_sharpe = perm_strat_perf.loc[key, 'oos_sharpe']
                
                diff_in_acc = (perm_acc - actual_acc) / actual_acc
                diff_in_sharpe = (perm_sharpe - actual_sharpe) / actual_sharpe
                
                var_differences_in_acc.append(diff_in_acc)
                var_differences_in_sharpe.append(diff_in_sharpe)
            
            # measure the mean and standard deviation of the decreases in accuracy and sharpe ratio
            mean_acc_diff = np.mean(var_differences_in_acc)
            mean_sharpe_diff = np.mean(var_differences_in_sharpe)
            sd_acc_diff = np.std(var_differences_in_acc)
            sd_sharpe_diff = np.mean(var_differences_in_sharpe)
            
            # Record mean difference in accuracy/sharpe
            model_acc_differences.append(mean_acc_diff)
            model_acc_diff_sd.append(sd_acc_diff)
            model_sharpe_differences.append(mean_sharpe_diff)
            model_sharpe_diff_sd.append(sd_sharpe_diff)
            
            
        # Store all arrays in a dict
        model_feat_import_dict['mean_acc_differences'] = model_acc_differences
        model_feat_import_dict['sd_acc_differences'] = model_acc_diff_sd
        model_feat_import_dict['mean_sharpe_differences'] = model_sharpe_differences
        model_feat_import_dict['sd_sharpe_differences'] = model_sharpe_diff_sd
        
        # Store model level dict into master dict 
        feature_importances[key] = model_feat_import_dict
            
    return feature_importances

def build_nn(architecture, input_shape, lr=0.001):
    
    
    """
    Builds a net according to specified architecture and activation function.
    
    Parameters 
    ----------
    architecture: an iterable containing the number of units in each layer. The length of the iterable determines the number of layers to add.
    activation: the activation function to use for each hidden_layer, 'elu' by default.
    
    Returns
    ----------
    model: a keras model object.
    """
    
    model = keras.models.Sequential()
    
    model.add(keras.Input(shape=input_shape))
    batch_standardize = False         
    
    n_layers = len(architecture)
    if n_layers >= 2:
        batch_standardize=True # for nets with two or more layers we apply batch normalization to avoid exploding/vanishing gradients
        
    for units in architecture:
        model.add(keras.layers.Dense(units, 
                                     activation='elu', 
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=keras.regularizers.l2(0.01)))
        if batch_standardize:
            model.add(keras.layers.BatchNormalization())
        
            
    model.add(keras.layers.Dense(1, activation='sigmoid'))
        
    optimizer=keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def fit_nets(X_train, y_train, X_val, y_val, callbacks, architecture=[64], lr=0.001, epochs=100):

    
    """

    Fits a net for each target variable.
    
    Parameters
    ----------
    X_train: the training feature set. 
    y_train: the training target set. 
    X_val:the validation feature set.
    y_val: the validation target set.
    callbacks: an iterable containing callbacks.
    architecture: an iterable containing the number of units in each hidden layer .
    lr: the learning rate for Nadam optimization.
    epochs: the number of training epochs. Doesn't really matter as long as it's large enough since 
        early stopping is used.

    Returns
    ----------
    models: a dict of KerasClassifiers indexable by target variable name.
    histories: a dict of training histories indexable by target variable name.
    """

    models = {}
    histories = {}
    
    for col in y_train.columns:
        
        model = keras.wrappers.scikit_learn.KerasClassifier(build_fn = lambda: build_nn(architecture=architecture, 
                                                                                                        input_shape=[len(X_train.columns)],
                                                                                                       lr=lr),
                                                           epochs=epochs,
                                                           batch_size=32)
        
        fit = model.fit(X_train, y_train[col], validation_data=(X_val, y_val[col]), callbacks=callbacks)
        
        models[col] = model
        histories[col] = fit
    
    return models, histories

def plot_histories(histories):
    
    """

    Plots the training history of a neural network. Must have metrics=['accuracy'].

    Parameters
    ----------
    histories: a dict of training histories indexable by target variable name.

    Returns
    ----------
    None


    """

    for key in histories:
        history = histories[key].history
        val_loss = history['val_loss']
        val_acc = history['val_accuracy']
        epochs = len(val_loss)
        lr = history['lr']
        history_df = pd.DataFrame({'epochs':epochs,'lr': lr,'val_loss': val_loss, 'val_accuracy':val_acc})
        
        print(key + f'loss plot over {epochs} iterations')
        history_df[['val_loss', 'val_accuracy']].plot()
        plt.xlabel('Iteration')
        plt.show()

    return None

def write_to_excel(df, sheet_name, path='Results.xlsx'):

    """

    Writes a DataFrame to an excel file.

    Parameters
    ----------
    df: a DataFrame
    sheet_name: the name of the sheet to write to
    path: the path to the excel workbook (default='Results.xlsx').

    Returns
    ----------
    None

    """

    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name=sheet_name)

    return None

def get_sklearn_model_params(models):

    """

    Extracts model parameters for each model in a dict of models, indexable by target variable name.

    Parameters
    ----------
    models: a dict of sklearn models indexable by target variable name.

    Returns
    ----------
    a DataFrame of model parameters 

    """

    params_df = pd.DataFrame()

    for key in models:
        model = models[key]
        try:
            params = model.best_params_ # for other kinds of models
        except:
            params = model.get_params() # for nets
        
        params_tmp_df = pd.DataFrame([params])
        params_df = pd.concat([params_df, params_tmp_df])
        
    params_df.insert(0, 'target', ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd'])

    return params_df

def fit_all_kinds_of_nets(layers, nodes, node_config='constant', first_layer_nodes=256):

    """

    Fits neural nets for every combination of the number of neurons (per layer) and the number of hidden layers.
    Nets are evaluated and results are saved.

    Parameters
    ----------
    layers: an iterable with elements indicating the number of hidden layers to create.
    nodes: an iterable with the number of nodes (per layer) to use.
    node_config: the method of congifuring nodes {'constant', 'halving'} (default='constant'). If node_config='constant', each layer has the same number of nodes.
        if node_config='halving', each subsequent layer is half of its predecessor. The number of nodes in the first layer can be specified. 
        By default, halving starts with 256 nodes in the first layer.
    first_layer_nodes: the number of nodes in the first hidden layer of the network. Only used if node_config is set to 'halving'. 

    Returns
    ----------
    None

    """

    for n_nodes in nodes:
        for n_layers in layers:
            
            # Constant configuration
            if node_config='constant':
                # defines neural network architecture
                arch = [n_nodes for layer in range(n_layers)]
                
                # fit nets 
                nn_models, nn_model_hists = fit_nets(X_train, y_train, X_val, y_val, callbacks=[early_stopping_cb, reduce_lr_cb], architecture=arch, epochs=300)
            
            elif node_config='halving':
                
                x = first_layer_nodes
                arch = [x]

                # if the number of layers is goe to 2, halve the nodes of each layer after the first
                if n_layers >= 2:
                    for layer in range(2, n_layers+1):
                        x = x / 2
                        arch.append(x)

                nn_models, nn_model_hists = fit_nets(X_train, y_train, X_val, y_val, callbacks=[early_stopping_cb, reduce_lr_cb], architecture=arch, epochs=300)


            # get params
            params = get_sklearn_model_params(nn_models)

            # evaluate 
            model_perf, strat_perf = evaluate_models(nn_models, X_train, y_train, X_test, y_test, strategy_train, strategy_test)

            model_name = 'NN ' + str(n_layers) + 'x' + str(n_nodes)

            model_perf.insert(1, 'model', model_name)
            strat_perf.insert(1, 'model', model_name)

            perf = pd.merge(model_perf, strat_perf, on=['target', 'model'])

            # save and write

            save_models(nn_models, model_name + '_')

            write_to_excel(pd.DataFrame(params), sheet_name= model_name +  ' Params')
            write_to_excel(perf, sheet_name= model_name + ' Performance')

    return None

    