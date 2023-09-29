#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:28:29 2023

@author: samuelahuno
"""
import numpy as np
import matplotlib.pyplot as plt



############################################################
## Step 1#
############################################################
# function to calculate likelihood for bernoulli
def logLikelihood(theta, x):
    # cal the log likelihood of each observation in the samples collected
    cal_llh = np.log(theta**(x) * (1-theta)**(1-x))
    tlld = np.sum(cal_llh)# cal the total likleihood
    return tlld

# function to calculate
def mle_Binom(X_samples, thetas):
    loglikelihood_single_theta = [logLikelihood(theta=t, x=X_samples) for t in thetas]
    # mle_val=thetas[np.argmax(likelihood_single_theta)] #get the maximum likelihood estimate
    return np.array(loglikelihood_single_theta)

# test the functions
true_params_Bern = 0.6
# mle_Binom(X_samples=np.random.binomial(n=1, p=true_params_Bern, size=100), thetas=np.linspace(start=0, stop=1, num=100))


############################################################
## Step 2#
############################################################
# how does the likelihood plot changes as sample size changes 
Bern_Nsamples = np.linspace(start=100, stop=1000, num=100, dtype=int)
response_Bernoulli = np.random.binomial(n=1, p=0.6, size=100)
possible_thetas = np.linspace(start=0.001, stop=1, num=100)
possible_thetas=possible_thetas[possible_thetas != 0] #remove 

def Bernoulli_optim(Bern_stepSize, rand_sets_thetas):
    for n in Bern_stepSize:
        response_Bernoulli = np.random.binomial(n=1, p=true_params_Bern, size=n) #generate samples from Binom Distri
        mle_out_Binom = mle_Binom(X_samples=response_Bernoulli, thetas=rand_sets_thetas) #cal lld of specific theta
        max_theta_Binom = rand_sets_thetas[np.argmax(mle_out_Binom)] #which theta gave us max lld
        #plot lld vrs thetas
        print(max_theta_Binom)
        fig, ax = plt.subplots()
        ax.plot(rand_sets_thetas, mle_out_Binom)
        # ax.scatter(max_theta_Binom, np.max(mle_out_Binom), color="red", label="MLE")
        ax.vlines(x=max_theta_Binom, ymin=np.min(mle_out_Binom), ymax=np.max(mle_out_Binom),
                  linestyles="dashed", color="red", label="MLE")
        ax.set_title(f'Bernoulli dist n = {n}')
        plt.xlabel("parameter(theta)")
        plt.ylabel("log likelihoods")
        plt.legend()
        plt.show()


#test function
# Bernoulli_optim(Bern_stepSize=Bern_Nsamples, rand_sets_thetas=possible_thetas)





############################################################
## Step #
############################################################
# how does sample size affect the mle
# plot number of samples and mle
Bern_Nsamples = np.linspace(start=100, stop=10000, num=100, dtype=int)
response_Bernoulli = np.random.binomial(n=1, p=0.6, size=100)
possible_thetas = np.linspace(start=0, stop=1, num=100)
possible_thetas=possible_thetas[possible_thetas != 0] #remove 
# result_theta = np.ma.array(possible_thetas.copy(), mask=possible_thetas.copy())

beta_for_mle_holder = []

def Bernoulli_optim_nSamples(Bern_stepSize, rand_sets_thetas): 
    for n in Bern_stepSize:
        response_Bernoulli = np.random.binomial(n=1, p=0.6, size=n)
        mle_out_Binom = mle_Binom(X_samples=response_Bernoulli, thetas=rand_sets_thetas) #cal lld of specific theta
        max_theta_Binom = rand_sets_thetas[np.argmax(mle_out_Binom)] #which theta gave us max lld
        beta_for_mle_holder.append(max_theta_Binom)
    fig, ax = plt.subplots()
    ax.plot(Bern_stepSize, beta_for_mle_holder, label = "MLE theta")
    ax.set_title('Bernoulli dist nSamples vrs MLE')
    ax.hlines(y=0.6, xmin=min(Bern_stepSize), xmax=max(Bern_stepSize), linestyles="dashed", color="red", label="MLE")
    plt.xlabel("nSamples")
    plt.ylabel("MLE")
    plt.show()


Bernoulli_optim_nSamples(Bern_stepSize=Bern_Nsamples, rand_sets_thetas=possible_thetas)


#Notes:
# At some large sample sizes the mle is close to true parameter of the binomial distribution. But the there is still varibaility in in the mle vrs nSamples
# now say, you don't know the true parameter (p) of this particualr distribution. we still need to find it somehow
# this is where the calculus ome  to play.
# the goal is to find the global minimum of any functin(imaginary or real). 
# we will use gradients on the curves of the function to find our way. the gradient is the rate of change of a function 
# mostly from an arbitary start point to the most probably point on the curve thay curresponds to global minimum