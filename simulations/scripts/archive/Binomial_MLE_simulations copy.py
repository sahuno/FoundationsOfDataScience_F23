#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:28:29 2023

@author: samuelahuno
"""
import numpy as np
import matplotlib.pyplot as plt
# calculate likelihood for bernoulli


def likelihood(theta, x):
    # cal the likelihood of each observation
    cal_lik = np.vectorize(lambda xi: theta**(xi) * (1-theta)**(1-xi))
    # cal the total likleihood
    tlk = np.prod(cal_lik(x))
    return tlk


# define values of x to be used for calculating
x_values = np.array([1, 0, 1, 0, 1])
likelihood(theta=0.2, x=x_values)

# now simulate for several theta. but how do you find these theta

# X_1 to X_n is fixed
# what is best theta


def mle(X, thetas):
    cal_lhd = np.vectorize(
        lambda single_thetas: likelihood(theta=single_thetas, x=X))
    likelihood_single_theta = cal_lhd(thetas)
    # mle_val=thetas[np.argmax(likelihood_single_theta)] #get the maximum likelihood estimate
    return likelihood_single_theta


possible_thetas = np.linspace(start=0, stop=1)
mle_out = mle(X=x_values, thetas=possible_thetas)

# likelihood(theta=0.5, x=x_values)
# likelihood(theta=0.1, x=x_values)

# which theta maximizes the function?
max_theta = possible_thetas[np.argmax(mle_out)]

# max_theta
# np.argmax(mle_out)
# np.max(mle_out)

fig, ax = plt.subplots()
ax.plot(possible_thetas, mle_out)
ax.scatter(max_theta, np.max(mle_out), color="red", label="MLE")
ax.vlines(x=max_theta, ymin=0, ymax=np.max(mle_out),
          linestyles="dashed", color="red", label="MLE")
ax.set_title("mle of possible thetas for bernoulli distribution")
plt.xlabel("thetas")
plt.ylabel("likelihoods")
plt.show()


#######################
##############################
# how does sample size affect the mle
Bern_Nsamples = np.linspace(start=100, stop=1000, num=100, dtype=int)
response_Bernoulli = np.random.binomial(n=1, p=0.6, size=100)


def Bernoulli_optim(Bern_stepSize, rand_sets_thetas):
    for n in Bern_stepSize:
        print(n)
        # print(Bern_stepSize[n])
        response_Bernoulli = np.random.binomial(n=1, p=0.6, size=n)
        mle_out_B = mle(X=response_Bernoulli, thetas=rand_sets_thetas)
        max_theta_B = rand_sets_thetas[np.argmax(mle_out_B)]
        fig, ax = plt.subplots()
        ax.plot(rand_sets_thetas, mle_out_B)
        ax.scatter(max_theta_B, np.max(mle_out_B), color="red", label="MLE")
        ax.vlines(x=max_theta_B, ymin=0, ymax=np.max(mle_out_B),
                  linestyles="dashed", color="red", label="MLE")
        ax.set_title(f'Bernoulli dist n = {n}')
        plt.xlabel("parameter(theta)")
        plt.ylabel("likelihoods")
        plt.show()


Bernoulli_optim(Bern_stepSize=Bern_Nsamples, rand_sets_thetas=possible_thetas)

# At some large sample sizes the mle is close to zero. does it mean the it's not in the set of thetas i used for my simulation?


# now say, you don't know the true parameter (p) of this particualr distribution. we still need to find it somehow
# this is where the calculus ome  to play.
# the goal is to find the global minimum of any functin(imaginary or real). 
# we will use gradients on the curves of the function to find our way. the gradient is the rate of change of a function 
# mostly from an arbitary start point to the most probably point on the curve thay curresponds to global minimum

Bern_Nsamples = np.linspace(start=100, stop=10000, num=100, dtype=int)
possible_thetas = np.linspace(start=0, stop=1, num=1000)
beta_for_mle_holder = []

def Bernoulli_optim(Bern_stepSize, rand_sets_thetas): 
    for n in Bern_stepSize:
        #print(n)
        # print(Bern_stepSize[n])
        response_Bernoulli = np.random.binomial(n=1, p=0.6, size=n)
        mle_out_B = mle(X=response_Bernoulli, thetas=rand_sets_thetas)
        max_theta_B = rand_sets_thetas[np.argmax(mle_out_B)]
        beta_for_mle_holder.append(max_theta_B)
    fig, ax = plt.subplots()
    ax.plot(Bern_stepSize, beta_for_mle_holder)
    ax.set_title(f'Bernoulli dist nSamples vrs MLE')
    ax.hlines(y=0.6, xmin=min(Bern_stepSize), xmax=max(Bern_stepSize), linestyles="dashed", color="red", label="MLE")
    plt.xlabel("nSamples")
    plt.ylabel("MLE")
    plt.show()


Bernoulli_optim(Bern_stepSize=Bern_Nsamples, rand_sets_thetas=possible_thetas)
