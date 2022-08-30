#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:49:50 2022

@author: s2113337

load pickle file of no prod energies 

do sklearn models on it?

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['axes.titlesize'] = 35
plt.rcParams['axes.titleweight'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.labelsize'] = 35 
plt.rcParams['lines.markersize'] = 2.5
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

#%%

path = '/home/s2113337/MAC-MIGS/IBM/all_data/'
data_mp2 = pd.read_pickle(path+'noprod_energies.pkl')

data_hf = pd.read_pickle(path+'noprod_energies_hf.pkl')

# linear regressor

def LR(data, PCA=True, plot=True):
    X = data.drop(columns = 'energy')
    y = data[['energy']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=3)

    scale = StandardScaler()
    X_tr = scale.fit_transform(X_train)
    X_te = scale.transform(X_test)

    scaley = StandardScaler()
    y_tr = scaley.fit_transform(y_train)
    y_te = scaley.transform(y_test)   

    
    lr = LinearRegression()
    folds = KFold(n_splits = 5, shuffle = True, random_state = 3)
    scores = cross_validate(lr, X_tr, y_tr, scoring=('r2', 'neg_mean_squared_error'), cv=folds)
    cv_MSE = -1*np.mean(scores['test_neg_mean_squared_error'])
    cv_r2 = np.mean(scores['test_r2'])

    lr.fit(X_tr, y_tr)

    y_pred = lr.predict(X_te)

    # mse
    print('Cross validation scores ')
    print("Root mean squared error: %.2f" % np.sqrt(cv_MSE))
    # r squared
    print("R squared score: %.2f" % cv_r2)

    if plot:
        plt.scatter(y_pred, y_te)
        plt.xlabel('Predicted ')
        plt.ylabel('Measured ')
        lims = [
                np.min([plt.xlim(), plt.ylim()]),  # min of both axes
                np.max([plt.xlim(), plt.ylim()]),  # max of both axes
                ]
        plt.plot(y_te, y_te, 'r', alpha=0.75, zorder=0)
        plt.title('Predicted v Measured ')
        plt.show()

        residual = y_te - y_pred

        plt.scatter(y_pred, residual)
        plt.title('Resiudal Plot for ')
        plt.xlabel('Predicted ')
        plt.ylabel('Residuals')
        y_zero = np.zeros(len(residual))
        x = np.linspace(np.min(y_pred),np.max(y_pred),len(residual))
        plt.plot(x, y_zero, 'r', alpha=0.75, zorder=0)
        plt.show()
        
    indep_r2 = lr.score(X_te, y_te)
    indep_MSE = mean_squared_error(y_te, y_pred)
    
    print('Independent test scores ')

    # mse
    print("Root mean squared error: %.2f" % np.sqrt(indep_MSE))
    # r squared
    print("R squared score: %.2f" % indep_r2)
    
    return [np.sqrt(indep_MSE), indep_r2]

#%% run for different methods 

mp2_scores = LR(data_mp2)

hf_scores = LR(data_hf)

