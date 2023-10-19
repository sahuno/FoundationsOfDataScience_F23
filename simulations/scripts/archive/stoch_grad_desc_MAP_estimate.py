#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:14:33 2023

@author: ahunos
implement stochastic gradient descent
"""

#this is  it. the beats are beta_1 and beta_not
import numpy as np

def compute_cost(A_data, y_responses, beta):
    m = len(y_responses)
    predictions = A_data.dot(beta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y_responses))
    return cost

def gradient_descent(A_data, y_responses, beta, learning_rate, num_iterations, tolerance=1e-6):
    m = len(y_responses)
    cost_history = []

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
    A_data = np.vstack([A_data, np.ones(len(A_data))]).T
    beta = np.zeros(A_data.shape[1])
    
    beta, _ = gradient_descent(A_data, y_responses, beta, learning_rate, num_iterations)
    return beta

# Test
A_data = np.array([1, 2, 3, 4, 5])
y_responses = np.array([5.5, 7, 8.5, 10, 11.5])
beta_estimated = linear_regression_gd(A_data, y_responses)
print(beta_estimated)
