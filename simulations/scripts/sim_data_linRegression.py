#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:39:20 2023

@author: ahunos
#simulate data linear regression
"""


import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)


def rand_X_y_LR(nsamples=None, nfeatures=None):
    
        # Define the number of samples and features
        num_samples = nsamples
        num_features = nfeatures
        
        # Generate a random design matrix X with values between 0 and 1
        X = np.random.rand(num_samples, num_features)
        
        # Generate random coefficients for the features
        true_coefficients = np.random.rand(num_features)
        
        # Generate random noise for the target variable
        noise = 0.1 * np.random.randn(num_samples)
        
        # Calculate the target variable y using a linear combination of X and coefficients
        y = np.dot(X, true_coefficients) + noise
        
        # Now, X is your 1000 by 24 design matrix, and y is the corresponding target variable
        return X, y



#test run 
# X_data, y_data = rand_X_y_LR(nsamples=1000, nfeatures=24)

# X_dataSGD, y_data_SGD = rand_X_y_LR(nsamples=1000, nfeatures=24)