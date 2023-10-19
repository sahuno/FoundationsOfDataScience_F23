#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:29:57 2023

@author: ahunos
"""

import numpy as np

def bayesian_mixture_generative_process(K, n, means, covariances, proportions):
    """
    A generative process for a Bayesian mixture model.
    
    Parameters:
    - K: Number of clusters
    - n: Number of observations
    - means: List of means for each cluster
    - covariances: List of covariance matrices for each cluster
    - proportions: Proportions of each cluster
    
    Returns:
    - x: Generated observations
    - z: Assignments of observations to clusters
    """
    
    # Ensure that the provided lists have the correct lengths
    assert len(means) == len(covariances) == K
    assert len(proportions) == K
    
    z = np.random.choice(K, size=n, p=proportions)  # Generate cluster assignments based on proportions
    x = np.array([np.random.multivariate_normal(means[zi], covariances[zi]) for zi in z])  # Generate observations
    
    return x, z

# Example usage:

# Number of clusters
K = 2

# Number of observations
n = 100

# Means for each cluster
means = [np.array([0, 0]), np.array([5, 5])]

# Covariance matrices for each cluster
covariances = [np.array([[1, 0], [0, 1]]), np.array([[1, 0.5], [0.5, 1]])]

# Proportions of each cluster
proportions = [0.5, 0.5]

x, z = bayesian_mixture_generative_process(K, n, means, covariances, proportions)
print("First 5 generated observations:", x[:5])
print("First 5 cluster assignments:", z[:5])
