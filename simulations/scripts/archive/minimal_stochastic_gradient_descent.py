#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:08:59 2023

@author: ahunos
"""

#implement minimal stochastic gradient
import numpy as np
from sklearn.model_selection import train_test_split


nsamples = 1000

mu1, sigma1 = 0, 0.1
mu2, sigma2 = 1, 2
muY, sigmaY = 0, 1


X1 = np.random.normal(mu1, sigma1, nsamples)
X2 = np.random.normal(mu2, sigma2, nsamples)
Y = np.random.normal(muY, sigmaY, nsamples)


X_data = np.stack((X1, X2), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.33, random_state=42)


#possible lambda estimates
prior_lambdas = np.linspace(start=0.1, stop = 10)


B_hat_MAP = 