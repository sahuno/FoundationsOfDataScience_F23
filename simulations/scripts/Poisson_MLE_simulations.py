#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 18:31:46 2023

@author: samuelahuno
"""

import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt


# poison distribution mle
def lld_poisson(lmda, X):
    lld = - lmda - gammaln(X + 1) + X*np.log(lmda)
    # tlld = np.sum(lld) 
    return np.sum(lld) 


def mle_pois(mle_lmda, mle_X):
    loglikelihood_single_lmda=np.array([lld_poisson(lmda=l, X=mle_X) for l in mle_lmda])
    return np.array(loglikelihood_single_lmda)



##run test
# rand_pois = np.random.poisson(lam=1, size=100) #generate random data from pois 
# lld_poisson(lmda=5, X=rand_pois) #compute log likelihood for single possible lamda
# random_lamdas=np.linspace(start=1, stop=100, dtype=int)
# np.random.shuffle(random_lamdas)
# mle_pois_run = mle_pois(mle_lmda=random_lamdas, mle_X=rand_pois) #what's the mle for data with true lamda = 1 
# random_lamdas[np.argmax(mle_pois_run)]

#any effect on ability to reach k as you increase sample size?

true_lamda_for_func = 5
Nsamples_pois_rand = np.linspace(start=100, stop=100000, num=100, dtype=int)
#pois_random = np.random.poisson(lam=true_lamda_for_func, size=100)
random_lamdas=np.linspace(start=1, stop=100, dtype=int)

#np.random.poisson(lam=true_lamda_for_func, size=100) #generate random data from pois 

#2 questions. 
#i. is the mle truly the sample mean?

def Pois_optim_nSamples(nIter=1,minSampleSize=5, maxSampleSize=100, sampleStepSize=5 ,true_lamda = 5): 
    Sample_sizes = np.arange(start=minSampleSize, stop=maxSampleSize+sampleStepSize, step=sampleStepSize) #set sample sets
    random_lamdas=np.linspace(start=1, stop=100, dtype=int) #generate random lambda values
    
    for _ in range(nIter):
        lamda_holder = []
        sampleMean_holder = []
        sampleVar_holder = []
        for n in Sample_sizes:
            random_pois_responses = np.random.poisson(lam=true_lamda, size=n)
            sample_mean=np.mean(random_pois_responses)
            sample_var=np.var(random_pois_responses)
            sampleMean_holder.append(sample_mean)
            sampleVar_holder.append(sample_var)
            mle_out_pois = mle_pois(mle_X=random_pois_responses, mle_lmda=random_lamdas) #cal lld of specific theta
            max_lamda_pois = random_lamdas[np.argmax(mle_out_pois)] #which theta gave us max lld
            lamda_holder.append(max_lamda_pois)
        fig, ax = plt.subplots()
        ax.plot(Sample_sizes, lamda_holder,label="MLE lambda")
        ax.plot(Sample_sizes, sampleVar_holder, label="Sample Var")
        ax.plot(Sample_sizes, sampleMean_holder, label="Sample Mean")
        ax.set_title(f'Poisson Dist, true lamda = {true_lamda} iter={_}')
        ax.hlines(y=true_lamda, xmin=min(Sample_sizes), xmax=max(Sample_sizes), linestyles="dashed", color="red", label="TRUE MLE")
        # ax.hlines(y=true_lamda_for_func, xmin=min(Sample_sizes), xmax=max(Sample_sizes), linestyles="dashed", color="red", label="MLE")
        # ax.hlines(y=true_lamda_for_func, xmin=min(Sample_sizes), xmax=max(Sample_sizes), linestyles="dashed", color="red", label="MLE")
        plt.xlabel("nSamples")
        plt.ylabel("measurement")
        plt.legend()
        plt.show()
    return lamda_holder, sampleMean_holder,sampleVar_holder
    #nIter+=1
#i=6
# for different sample size run algorithms
# lamdas_out,sMeans_out,sVar = Pois_optim_nSamples(true_lamda=i, nIter=1)
for i in range(1,10):
    lamdas_out,sMeans_out,sVar = Pois_optim_nSamples(true_lamda=i, nIter=1)


###simulate variance and mean  of  poisson
#vary the true lambda
#sample size 50 - 100, step size 5
#
def Pois_monte_carlo_sim(Sample_sizes, rand_sets_lamda, minSample=10, maxSample= 100, sampleStepSize = 5): 
    lamda_holder = []
    sampleMean_holder = []
    sampleVar_holder = []
    # Nsamples_pois_rand = np.linspace(start=10, stop=100, num=10, dtype=int)
Nsamples = np.linspace(start=10, stop=100, num=10, dtype=int)
    for n in Nsamples:
        print(n)
        
        random_pois_responses = np.random.poisson(lam=true_lamda, size=n)
        sample_mean=np.mean(random_pois_responses)
        sample_var=np.var(random_pois_responses)
        sampleMean_holder.append(sample_mean)
        sampleVar_holder.append(sample_var)
        mle_out_pois = mle_pois(mle_X=random_pois_responses, mle_lmda=rand_sets_lamda) #cal lld of specific theta
        max_lamda_pois = rand_sets_lamda[np.argmax(mle_out_pois)] #which theta gave us max lld
        lamda_holder.append(max_lamda_pois)
    