#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:18:41 2023

@author: ahunos
"""

import numpy as np
from sklearn.model_selection import train_test_split


def add_bias_term(A_data):
    ones_column = np.ones((len(A_data), 1))
    return np.hstack([A_data, ones_column])

# Modify compute_cost to incorporate Bayesian priors
def compute_cost(A_data, y_responses, beta, lambda_reg):
    m = len(y_responses)
    predictions = A_data.dot(beta)
    reg_term = (lambda_reg / 2) * np.sum(np.square(beta))
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y_responses)) + reg_term
    return cost

# Modify gradient descent for Bayesian regularization and to use stochastic approach
def stochastic_gradient_descent(A_data, y_responses, beta, learning_rate, num_iterations, lambda_reg, tolerance=1e-6):
    m = len(y_responses)
    cost_history = []

    for iteration in range(num_iterations):
        # Randomly select one data point
        idx = np.random.randint(0, m)
        xi = A_data[idx:idx+1]
        yi = y_responses[idx:idx+1]
        
        # Calculate gradient using the selected data point
        gradient = xi.T.dot(xi.dot(beta) - yi) + lambda_reg * beta
        
        # Update beta
        beta -= learning_rate * gradient

        # Compute cost
        cost = compute_cost(A_data, y_responses, beta, lambda_reg)
        cost_history.append(cost)

        # Check for convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            break

    return beta, cost_history


# Modify linear_regression_gd function to use the stochastic gradient descent and include lambda_reg
def linear_regression_gd(A_data, y_responses, learning_rate=0.01, num_iterations=1000, lambda_reg=1/2000):  # 1/2σ^2
    A_data = add_bias_term(A_data)
    beta = np.zeros(A_data.shape[1])
    beta, _ = stochastic_gradient_descent(A_data, y_responses, beta, learning_rate, num_iterations, lambda_reg)
    return beta


#now fit the data
file_path = "/Users/ahunos/Downloads/methyl_and_DE_dt_Gapdh_ENSMUSG00000057666_1.tsv"
data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
# Extract the last column (Y variable)
Y_variable = data[:, -1]
# Extract the rest of the columns (X variables)
X_variables = data[:, :-1]

linear_regression_gd(A_data=X_variables, y_responses=Y_variable)


