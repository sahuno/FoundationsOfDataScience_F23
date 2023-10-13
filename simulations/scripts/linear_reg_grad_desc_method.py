#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:25:02 2023

@author: ahunos
"""

#this is  it. the beats are beta_1 and beta_not
import numpy as np
from sim_data_linRegression import rand_X_y_LR #function to simulate lienar regression
from sklearn.model_selection import train_test_split


def add_bias_term(A_data):
    ones_column = np.ones((len(A_data), 1))
    return np.hstack([A_data, ones_column])


#def function to compute cost
def compute_cost(A_data, y_responses, beta):
    m = len(y_responses)
    predictions = A_data.dot(beta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y_responses))
    return cost


#define grad desc func
def gradient_descent(A_data, y_responses, beta, learning_rate, num_iterations, tolerance=1e-6):
    m = len(y_responses)
    cost_history = [] #keep track of how i'm doing for each iteration

    for iteration in range(num_iterations):
        # Calculate gradient
        gradient = (1 / m) * A_data.T.dot(A_data.dot(beta) - y_responses)
        
        # Update beta
        beta -= learning_rate * gradient #this should be positive cos we doing gradient accent

        # Compute cost
        cost = compute_cost(A_data, y_responses, beta)
        cost_history.append(cost)

        # Check for convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            break

    return beta, cost_history

# Using the gradient descent for linear regression
def linear_regression_gd(A_data, y_responses, learning_rate=0.01, num_iterations=1000):
    # Adding the bias term to the input
    A_data = add_bias_term(A_data)
    beta = np.zeros(A_data.shape[1])
    
    beta, _ = gradient_descent(A_data, y_responses, beta, learning_rate, num_iterations)
    return beta

# Test the function
# A_data = np.array([1, 2, 3, 4, 5])
# y_responses = np.array([5.5, 7, 8.5, 10, 11.5])


#simulate datasets
X_data, y_data = rand_X_y_LR(nsamples=1000, nfeatures=15)

#split datasets to in and out samples
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)



#estimate betas using trainig data
beta_estimated_train = linear_regression_gd(A_data=X_train, y_responses=y_train)



    
##part 2:
    #test estimated betas on out samples
A_data_test = add_bias_term(X_test)
def prediction(Xdata, beta):
    y_pred = Xdata.dot(beta)
# cost_in_sample = compute_cost(A_data=add_bias_term(X_train), y_responses=y_train, beta=beta_estimated_train)
# cost_out_sample = compute_cost(A_data=add_bias_term(X_test), y_responses=y_test, beta=beta_estimated_train)

