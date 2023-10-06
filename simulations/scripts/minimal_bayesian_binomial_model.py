#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:25:59 2023

@author: ahunos
#implment a minimal bayseain bernollli model

"""

import numpy as np
#given this obervation with true parameter=0.6(we pretend we don't know this cos in the real worl we don't)

#define function to geerate random bernoulli data
true_param = 0.6
def gen_random_bern_data(sampleSize=5, true_param = 0.6):
    return np.random.binomial(n=1, p=true_param, size=sampleSize)

X_res = gen_random_bern_data(sampleSize=5, true_param = 0.6)
X_res

#func log-likehood    
def log_lik(theta, X):
    theta_i_logliklihood_for_data_Xi=np.log(theta)*(X) + np.log((1-theta))*(1-X)
    theta_i_logliklihood_for_data_Xi_to_Xn = np.sum(theta_i_logliklihood_for_data_Xi)
    return theta_i_logliklihood_for_data_Xi_to_Xn


#now run 
posssible_thetas=np.linspace(start=0.01, stop=0.99, num=100)
log_lik(theta=posssible_thetas[1], X=X_res) #calculate likelihood for single theta
#how about likelihood for all the possible thetas?
log_likel_for_thetas = [log_lik(theta=each_theta, X=X_res) for each_theta in posssible_thetas]
#then pick the best one
most_likel_theta_for_data = posssible_thetas[np.argmax(log_likel_for_thetas)]

#how does this deviate from true params?
mle_deviation = true_param - most_likel_theta_for_data
if np.abs(mle_deviation) > 0:
    print(f'estimated parameter {most_likel_theta_for_data} deviates from the true params {true_param} by {mle_deviation}')
#this is what we termed as mle
    

### part two. does number of damoles hacve any effect on the accuracy of the mle
#what happens as you incrase the sample size 
#does increasing the true paramter have an efect on mle
# is mle the same for every iteration of same number of sample and paramter?
# def sample_size_var():

### now assume an expert says the theta should be 0.3