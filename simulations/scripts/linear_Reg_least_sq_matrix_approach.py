#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:23:13 2023

@author: ahunos
"""

import numpy as np

##method 1: linear algebrae method
#from wes: this only works, x'x is invertible and there's no colinearity between the errors
def least_sq_estimator(A_data, y_responses, gamma_val=2):
# Direct least square regression
    ##add 1's to column of data matrix and transpose
    A_data = np.vstack([A_data, np.ones(len(A_data))]).T #book's approach, ensures output beta_1 followed by beta_zero
    # A_data = np.vstack([np.ones(len(A_data)), A_data]).T #inquiry, what happends if
    beta_hats = np.dot((np.dot(np.linalg.inv(np.dot(A_data.T,A_data)),A_data.T)),y_responses)
    if gamma_val is not None:
        return (beta_hats + (1 / (2*gamma_val)))
    else:
        return beta_hats


#Run function
# generate random x and y
x = np.linspace(0, 1, 101)
y = 1 + x + x * np.random.random(len(x))
y = y[:, np.newaxis]

beta_est = least_sq_estimator(A_data=x, y_responses=y)
#this estimates $\beta_not$ and beta_hat

beta_est2 = least_sq_estimator(A_data=x, y_responses=y)




######################################
#####method 2:
def compute_pred(X, y, beta_hat):
    y_pred = (X @ beta_hat)  #find predicted y - values
    return y_pred
    
    
def compute_cost(X, y, beta_hat):
    """
    Compute the cost function J for linear regression.
    this is objective function and what we want to minimize
    """
    m = len(y) #length of data  ~ nsamples
    #predictions = X.dot(beta_all) #find predicted y - values
    #predictions =  numpy.dot(X, theta) #also works
    #predictions = X @ theta #alternatives
    y_preds = compute_pred(X,y,beta_hat)
    square_errors = (y_preds - y) ** 2 #square errors
    sum_square_errors = np.sum(square_errors) #sum of square errors
    mean_square_errors =  1 / (2 * m) * sum_square_errors #mean sq errors
    return mean_square_errors




    




# def gradient_descent(X, y, theta, stepSize=0, num_iterations):
#     """
#     Perform gradient descent to learn theta.
#     """
#     m = len(y)
#     J_history = np.zeros(num_iterations)
    
#     #calculate beta 0
#     # Calculate means
#     y_bar = np.mean(y)
#     x_bar = np.mean(X, axis=0)  # this will give you a 10x1 vector of means for each column of X

#     # Compute beta_0
#     beta_0 = y_bar - np.dot(x_bar, beta_all)
    
    
#     for iter in range(num_iterations):
#         errors = (X.dot(theta) - y)
#         delta = (1/m) * X.T.dot(errors)
#         theta = theta - alpha * delta
#         J_history[iter] = compute_cost(X, y, theta)
    
#     return theta, J_history











# def gradient_descent(X, y, theta, alpha, num_iterations):
#     """
#     Perform gradient descent to learn theta.
#     """
#     m = len(y)
#     J_history = np.zeros(num_iterations)
    
#     for iter in range(num_iterations):
#         errors = (X.dot(theta) - y)
#         delta = (1/m) * X.T.dot(errors)
#         theta = theta - alpha * delta
#         J_history[iter] = compute_cost(X, y, theta)
    
#     return theta, J_history

# def linear_regression(X, y, alpha, num_iterations):
#     """
#     Run linear regression on input dataset using gradient descent.
#     """
#     # Add a column of ones to X to account for the intercept (theta_0)
#     X = np.column_stack((np.ones((X.shape[0], 1)), X))
#     theta = np.zeros((X.shape[1], 1))
    
#     theta, J_history = gradient_descent(X, y, theta, alpha, num_iterations)
    
#     return theta, J_history

# # Example usage:
# # Assume you have X_train (training data) and y_train (training labels) as numpy arrays
# # X_train = np.array([[...], [...], ...])
# # y_train = np.array([[...], [...], ...])

# # alpha = 0.01
# # num_iterations = 1500
# # theta, J_history = linear_regression(X_train, y_train, alpha, num_iterations)
