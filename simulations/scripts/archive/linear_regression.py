#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:59:21 2023

@author: ahunos
"""

#minimal linear regression
import numpy as np

def linear_regression_gradient_descent(X, y, learning_rate, num_iterations):
    # Initialize weights and bias to zeros
    num_samples, num_features = X.shape
    theta = np.zeros((num_features, 1))
    bias = 0
    
    for iteration in range(num_iterations):
        # Calculate predictions
        predictions = np.dot(X, theta) + bias
        
        # Calculate the error (mean squared error)
        error = (1 / (2 * num_samples)) * np.sum((predictions - y) ** 2)
        
        # Calculate the gradients
        d_theta = (1 / num_samples) * np.dot(X.T, (predictions - y))
        d_bias = (1 / num_samples) * np.sum(predictions - y)
        
        # Update weights and bias using gradient descent
        theta -= learning_rate * d_theta
        bias -= learning_rate * d_bias
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Error = {error}")
    
    return theta, bias

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Add a column of ones to X for the bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform linear regression
theta, bias = linear_regression_gradient_descent(X, y, learning_rate, num_iterations)

# Print the learned parameters
print("Learned parameters:")
print(f"Theta: {theta}")
print(f"Bias: {bias}")
