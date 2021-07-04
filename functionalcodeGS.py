#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 23:54:57 2021

@author: sristi
"""
import numpy as np
import pandas as pd
from scipy.stats import loguniform

import logging

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
import re
from matplotlib import pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

   
train_data = pd.DataFrame()
predict_data = pd.DataFrame()

def LabelEncoderMultiColumns(data, columns_list):
    """Encoding categorical feature in the dataframe

    Parameters
    ----------
    data: input dataframe 
    columns_list: categorical features list

    Return
    ------
    data: new dataframe where categorical features are encoded
    """
    labelencoder = LabelEncoder()
    for col in columns_list:
        data[col] = labelencoder.fit_transform(data[col])
    return data

def score(y_true, y_pred, **kwargs):
    return max(0, 100*r2_score(y_true, y_pred))
r2 = make_scorer(score, greater_is_better = False)


def KFoldCV(n_splits = 10):
    # k-fold cross validation
    kfold = KFold(n_splits=n_splits, shuffle=True)
    return kfold
    
def LogisticRegressionModel(X, y):
    logging.info('Started Grid Search for Logistic Regression...')
    
    # Add column for classification labels, to tell if the person is eligible for loan sanction or not
    X['Loan Eligibility'] = np.where(y>0, 1, 0)
    y = X['Loan Eligibility']
    X.drop('Loan Eligibility', axis = 1, inplace = True)
    
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = LogisticRegression(random_state=0, max_iter = 500)
        # define search space
#        space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
#        space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
        param_grid = [
                {'penalty': ['l2'], 'solver': [ 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg']},
                {'penalty': ['l1'], 'solver': ['liblinear', 'saga']},
        ]
    	# define search
        search = GridSearchCV(model, param_grid = param_grid, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        logging.info('acc = {}, est = {}, cfg = {}'.format(acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    logging.info('\nAccuracy: {} (Standard Deviation: {})'.format(np.mean(outer_results), np.std(outer_results)))
    logging.info('Stopped Grid Search for Logistic Regression')

    
def DecisionTreeModel(X, y):
    logging.info('Started Grid Search for Decision Tree...')
    
    # Add column for classification labels, to tell if the person is eligible for loan sanction or not
    X['Loan Eligibility'] = np.where(y>0, 1, 0)
    y = X['Loan Eligibility']
    X.drop('Loan Eligibility', axis = 1, inplace = True)
    
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = DecisionTreeClassifier(random_state=0)
        # define search space
#        space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
#        space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
        param_grid = [
                {'criterion': ['entropy', 'gini'], 'max_depth': np.arange(3, 15)}, 
                {'min_samples_leaf': np.arange(1,10)},
#                {'min_samples_split': np.arange(1,10)}
        ]
    	# define search
        search = GridSearchCV(model, param_grid = param_grid, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        logging.info('acc = {}, \n best score = {}, \n best params = {}, \n'.format(acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    logging.info('\nAccuracy: {} (Standard Deviation: {})'.format(np.mean(outer_results), np.std(outer_results)))
    logging.info(best_model)
    logging.info('Stopped Grid Search for Decision Tree')
    
        
def LinearRegressionModel(X, y):
    logging.info('Started Grid Search for Linear Regression...')
        
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = LinearRegression()
        # define search space
        space = dict()
        space['fit_intercept'] = [True, False]
        space['normalize'] = [True, False]
       
    	# define search
        search = GridSearchCV(model, space, scoring=r2, cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        logging.info('acc = {}, est = {}, cfg = {}'.format(acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    logging.info('\nAccuracy: {} (Standard Deviation: {})'.format(np.mean(outer_results), np.std(outer_results)))
    logging.info('Stopped Grid Search for Linear Regression')


def LassoRegressionModel(X, y):
    logging.info('Started Grid Search for Lasso Regression...')
        
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = Lasso(max_iter = 2000)
        # define search space
        space = dict()
        space['alpha'] = np.logspace(1e-5, 1)
        space['fit_intercept'] = [True, False]
        space['normalize'] = [True, False]
       
    	# define search
        search = GridSearchCV(model, space, scoring=r2, cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        logging.info('acc = {}, est = {}, cfg = {}'.format(acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    logging.info('\nAccuracy: {} (Standard Deviation: {})'.format(np.mean(outer_results), np.std(outer_results)))
    logging.info('Stopped Grid Search for Lasso Regression')


def SVRModel(X, Y):
    logging.info('Started Grid Search for Support Vector Regression...')
        
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = SVR()
        # define search space
        space = dict()
        space['C']= [0.1, 1]
        space['gamma'] = [1,0.1,0.001]
        space['kernel'] = ['rbf', 'sigmoid']
       
    	# define search
        search = GridSearchCV(model, space, scoring=r2, cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        logging.info('acc = {}, est = {}, cfg = {}'.format(acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    logging.info('\nAccuracy: {} (Standard Deviation: {})'.format(np.mean(outer_results), np.std(outer_results)))
    logging.info('Stopped Grid Search for Support Vector Regression')

def RandomForestModel(X, Y):
    logging.info('Started Grid Search for Random Forest Regression...')
        
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = RandomForestRegressor(random_state = 0)
        # define search space
        space = dict()
        space['n_estimators'] = [100,1000,10000]
        space['max_features'] = ["auto", "sqrt", "log2"]
        space['min_samples_split'] = [2,4,8]
        space['bootstrap'] = [True, False]
       
    	# define search
        search = GridSearchCV(model, space, scoring=r2, cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        logging.info('acc = {}, est = {}, cfg = {}'.format(acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    logging.info('\nAccuracy: {} (Standard Deviation: {})'.format(np.mean(outer_results), np.std(outer_results)))
    logging.info('Stopped Grid Search for Random Forest Regression')

def read_data(train_location, predict_location):
    global train_data
    train_data = pd.read_csv(train_location)
    global predict_data 
    predict_data = pd.read_csv(predict_location)

 
def missing_values(data):
    data['Dependents'].fillna(data['Dependents'].value_counts().idxmax(), inplace=True)
    #       For numeric columns
    data.fillna(data.mean(), inplace = True)
    
    #       For string columns
    nan_columns = data.isna().any()
    nan_columns = nan_columns[nan_columns == True]
    #       Replace missing value with the most common value of that column
    for column in nan_columns.iteritems():
        data[column[0]].fillna(data[column[0]].value_counts().idxmax(), inplace=True)
    return data
    
def categorical_values(data, test):
    # For columns having two categories
    
    data = LabelEncoderMultiColumns(data, ['Gender', 'Income Stability', 'Expense Type 1', 'Expense Type 2'])

    # For columns having multiple categories
    if test==True:
        data = onehotencoder.transform(data)
    else:
        onehotencoder.fit(data)
        data = onehotencoder.transform(data)
    column_names = onehotencoder.get_feature_names()
    # Remove 'onehotencoder' from column labels
    for index, name in enumerate(column_names):
        column_names[index] = re.sub(r'onehotencoder__x', '', name)
    data = pd.DataFrame(data, columns=column_names)
    return data
    
def normalization(data, test):   
    if test==True:
        data = pd.DataFrame(minmaxscaler.transform(data), columns = data.columns)
    else:
        minmaxscaler.fit(data)
        data = pd.DataFrame(minmaxscaler.transform(data), columns = data.columns)
    return data
    
    
def clean_train_data():
    global train_data
    # 1. Remove columns containing unimportant information
    train_data.drop(['Customer ID', 'Name', 'Property ID'], axis = 1, inplace = True)
    # 2. Replace any datapoints containing unwanted characters with NaN
    train_data.replace({r'[+=!~`:;?<>@#$%^&*]': None}, regex = True, inplace = True)
    # 3. Treat missing values
    train_data = missing_values(train_data)
    # 4. Seperate X_train and y_train for further pre-processing
    train_Y = train_data['Loan Sanction Amount (USD)']
    train_data = train_data.drop(['Loan Sanction Amount (USD)'], axis = 1)
    # 5. Deal with columns containing categorical values - Dummy variables
    global onehotencoder 
    onehotencoder = make_column_transformer((OneHotEncoder(), ['Location', 'Has Active Credit Card', 'Property Location', 'Profession', 'Type of Employment']), remainder='passthrough')

    train_data = categorical_values(train_data, False)
    # 6. Normalization
    global minmaxscaler
    minmaxscaler = MinMaxScaler()
    train_data = normalization(train_data, False)
    train_data['Loan Sanction Amount (USD)'] = train_Y
    del train_Y
    
    
def clean_predict_data():
    global predict_data
    # 1. Remove columns containing unimportant information
    predict_data.drop(['Customer ID', 'Name', 'Property ID'], axis = 1, inplace = True)
    # 2. Replace any datapoints containing unwanted characters with NaN
    predict_data.replace({r'[+=!~`:;?<>@#$%^&*]': None}, regex = True, inplace = True)
    # 3. Treat missing values
    predict_data = missing_values(predict_data)
    # 4. Deal with columns containing categorical values - Dummy variables
    predict_data = categorical_values(predict_data, True)
    # 6. Normalization
    predict_data = normalization(predict_data, True)
        
    
    
read_data('train.csv', 'test.csv')
clean_train_data()

y = train_data['Loan Sanction Amount (USD)']
X = train_data.drop(['Loan Sanction Amount (USD)'], axis = 1)

#LogisticRegressionModel(X, y)
#DecisionTreeModel(X, y)

train_data_copy = train_data.copy()
train_data_copy.drop(train_data_copy[train_data_copy['Loan Sanction Amount (USD)'] == 0].index, inplace = True)
y_copy = train_data_copy['Loan Sanction Amount (USD)']
X_copy = train_data_copy.drop(['Loan Sanction Amount (USD)'], axis = 1)
#LinearRegressionModel(X_copy, y_copy)
#assoRegressionModel(X_copy, y_copy)
#SVRModel(X, y)
RandomForestModel(X_copy, y_copy)
clean_predict_data()

# =============================================================================
# 
# train.plot.scatter('Customer ID', 'Income (USD)')
# train.fillna(train.mean(), inplace=True)
# train.plot.scatter('Customer ID', 'Income (USD)')
# 
# train.plot.scatter('Customer ID', 'Income Stability')
# train.fillna(test.mean(), inplace=True)
# train.plot.scatter('Customer ID', 'Income Stability')
# 
# train.plot.scatter('Customer ID', 'Current Loan Expenses (USD)')
# 
# 
# train.plot.scatter('Customer ID', 'Credit Score')
# 
# test.fillna(test.mean(), inplace=True)
# 
# =============================================================================























