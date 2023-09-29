#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:09:30 2023

@author: samuelahuno
"""

#homework 1. Regularized poisson regresssion
#simulation X and Y data sets 

#load libraries
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

nSamples=1000
Beta=3
X = np.random.normal(loc=0, scale=1, size=nSamples) 

#generate lambda values to use for poisson
lambda_vals = np.exp(Beta*X)
Y = poisson.rvs(mu=lambda_vals, size=nSamples) #generate responses
 

fig, ax = plt.subplots()
plt.scatter(X,Y)
ax.set_title("X ~ N(0,1) , Y ~ P(exp(Bx)) ; where B=3")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# class model(numb_samples, norm_mu, norm_sd):
#     def f(self):
#         print("first class")


        