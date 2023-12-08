#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ahunos
"""

#fnctions
# create bacthes
# Use the robinson monrow step size schedule
# add bias to X_dataa
# compute cost
#comupte gradient
#main stochatic grad descent for bmap

import numpy as np
from sklearn.model_selection import train_test_split

class SGD_BMAP():
    
    def __init__(self, Xdata, y_data, numBatches=5, nIterations=100, tol=1e-6, alpha=1e-3, prior_beta_var_lambda=10):
        self.X = Xdata
        self.y = y_data
        self.nB = numBatches
        self.rhoT = alpha
        self.nIter = nIterations
        self.prior_beta_var_lambda = prior_beta_var_lambda
        self.tol = tol
        self.nObserEntireDataset = len(Xdata)
        #print(f'lambda prior is {self.prior_beta_var_lambda}')
        
    def add_bias(self):
        '''
        #func to add bias to coeffients
        '''
        ones_column = np.ones((len(self.X), 1))
        return np.hstack([self.X, ones_column])
    
    def createBatches(self):
        """
        #define function to compute n batches
        """
        bSize = int(len(self.X) / self.nB) # what shld each batch look lke
        
        #create indexes of data to use for gradient descent
        dataInd = [np.random.choice(range(len(self.X)), size = bSize, replace = None) for _ in range(self.nB)] 
        
        # function to slice data
        def make_data_slices(x, y, idx):
            X_data_slice = x[idx]
            y_data_slice = y[idx]
            return {"X":X_data_slice, "Y":y_data_slice}
        
        #make batches of data
        data_slice = [make_data_slices(x = self.X, y= self.y, idx = ind) for ind in dataInd]
        return data_slice
    
    #function to compute y pred
    def pred_y(self, betas, X_batch):
        #print(betas)
        y_batch_predictions = np.dot(X_batch, betas)
        return y_batch_predictions
    
    #calculate the gradient
    def sgd_gradient(self, beta_sgd, lambda_val, X_batch, y_batch_pred, y_batch_true):
        bsize=len(X_batch)
        first_term = - (1/lambda_val) * beta_sgd
        error_term = (y_batch_true - y_batch_pred).reshape(-1, 1)
        second_term = (self.nObserEntireDataset/bsize) * np.sum(error_term * X_batch, axis=0)
        grad = first_term + second_term
        return grad
    
    def run_BMAP_SGD(self):
        #a, b, gamma = 1, 1, 0.75 #Rob-Monroe step size params
        rhoT = self.rhoT
        beta = np.zeros(self.X.shape[1])
        iter = 1
        beta_holder = [beta]
        print(f'using prior lambda as {self.prior_beta_var_lambda}')
        while iter < self.nIter:
            #get a bacth of data
            for batch_data in self.createBatches():
            # batch_data = self.createBatches(self.X, self.y)
    
                #cmpute predicted expectation
                y_pred = self.pred_y(betas=beta, X_batch=batch_data['X'])
                
                #compute sgd gradient
                gradient = self.sgd_gradient(beta_sgd = beta, lambda_val = self.prior_beta_var_lambda, X_batch = batch_data['X'], y_batch_pred = y_pred, y_batch_true = batch_data['Y'])
                
                #update betas
                beta = beta + (rhoT * gradient)
                
                beta_holder.append(beta) #keep track of betas
                
                #update Robinson and monroe step size
                #rhoT = 1/iter #Dave's book! 
                rhoT = 1/np.square(iter)
                #alternatively 
                #rhoT = 1 / (iter**0.5)
                #rhoT = a / (b + iter) ** gamma #other suggestions online
                
                #check for convergence with l2 norm strategy
                if iter > 1 and np.linalg.norm(beta_holder[-1] - beta_holder[-2]) <= self.tol:
                    break
                    print(f'converged at batch iter{iter}') #why is not printing?
                
                iter+=1
        return beta, beta_holder
    
    def compute_inSample_llhd(self, calculated_BMAP, sigma_for_llhd):
        #BMAP, _ = self.run_BMAP_SGD() #get SGD BMAP
        m = len(self.y)
        yPredictionsSelf = self.X.dot(calculated_BMAP)
        inSample_llhd = (-(1 / m)) * np.sum((1/ (2 * np.square(sigma_for_llhd))) * np.square(yPredictionsSelf - self.y))
        return inSample_llhd
    
    #compute cost of bmap for sible hyperparameter
    def compute_outSample_llhd(self, calculated_BMAP, XTEST, YTEST, sigma_for_llhd):
        #BMAP, _ = self.run_BMAP_SGD() #get SGD BMAP
        m = len(YTEST)
        predictions = XTEST.dot(calculated_BMAP)
        post_llhd = (-(1 / m)) * np.sum((1/ (2 * np.square(sigma_for_llhd))) * np.square(predictions - YTEST))
        return post_llhd
    



    
##########################################################
##########################################################
###### Generate random data to test the function

##test the dunction see if it works
np.random.seed(0)  # for reproducibility

# Generating random dataset of size 1024x24 for X
Xdata = np.random.rand(1024, 24)
 #normalize data, suggestion online
Xdata_in = (Xdata - np.mean(Xdata, axis=0)) / np.std(Xdata, axis=0)

# Generating random continuous y values
y_data_in = np.random.rand(1024)

# Split the data into training and temporary sets (70% train, 30% temp)
X_train, X_temp, y_train, y_temp = train_test_split(Xdata_in, y_data_in, test_size=0.3, random_state=42)
# Split the temporary set into validation and test sets (50% validate, 50% test from the temp set)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
###### Generate random data to test the function


#
#run for 3 hypeparameters
# Initialize and test the SGD_BMAP class
# hyperVals = np.arange(10, 100, 5)


# for i,v in enumerate(hyperVals):
#     model = SGD_BMAP(Xdata=X_train, y_data=y_train, alpha=0.01, numBatches=10, prior_beta_var_lambda=v)
#     beta_final, beta_history = model.run_BMAP_SGD()
#     print(f"Betas for hyperparam {v} :", beta_final)
#     model.compute_cost(calculated_BMAP=beta_final, XTEST=X_test, YTEST=y_test)


##########################################################
##########################################################
###### run model
#draw prior lambda from normal distribution
lambda_normal = np.arange(10, 1000, 5)
in_llhd_holder = dict()
out_llhd_holder = dict()

for i,v in enumerate(lambda_normal):
    model_SGD = SGD_BMAP(Xdata=X_train, y_data=y_train, alpha=0.01, numBatches=10, prior_beta_var_lambda=v)
    beta_final2, _ = model_SGD.run_BMAP_SGD()
    inSample_llhd = model_SGD.compute_inSample_llhd(calculated_BMAP=beta_final2, sigma_for_llhd=1)
    outSample_llhd = model_SGD.compute_outSample_llhd(calculated_BMAP=beta_final2, XTEST=X_test, YTEST=y_test, sigma_for_llhd=1)
    out_llhd_holder[v] = outSample_llhd 
    in_llhd_holder[v] = inSample_llhd
    #llhd_stacked = np.hstack((inSample_llhd, outSample_llhd))

#--out: -1.2311836348541705e+28-(-1.0765940022494813e+28)
    
    
#####################
###################
import matplotlib.pyplot as plt

indexes = list(out_llhd_holder.keys())
out_sample_llhdsData = list(out_llhd_holder.values())
in_sample_llhdsData = list(in_llhd_holder.values())

# Separate in-sample and out-of-sample likelihoods
#in_sample_llhds = [value[0] for value in llhd_holder.values()]
#out_sample_llhds = [value[1] for value in llhd_holder.values()]
#abs_diff_outIn =  np.absolute(out_sample_llhdsData) - np.absolute(in_sample_llhdsData)
#abs_diff_outIn =  np.subtract(out_sample_llhdsData) - np.subtract(in_sample_llhdsData)
abs_diff_outIn = np.absolute(np.array(out_sample_llhdsData) - np.array(in_sample_llhdsData))
#abs_diff_outIn = np.array(out_sample_llhdsData) - np.array(in_sample_llhdsData)


# Creating subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot in-sample LLHD
axs[0].plot(indexes, in_sample_llhdsData, color='blue')
axs[0].plot(indexes, abs_diff_outIn, color='black', linestyle='-', label='Absolute Difference')
axs[0].set_title('In-sample Log-Likelihood BMAP_LinRegr')
axs[0].set_xlabel('Lambda Hyperparameter')
axs[0].set_ylabel('In-sample LLHD')
axs[0].legend()

# Plot out-of-sample LLHD
axs[1].plot(indexes, out_sample_llhdsData, color='red')
axs[1].plot(indexes, abs_diff_outIn, color='black', linestyle='-', label='Absolute Difference')
axs[1].set_title('Out-of-sample Log-Likelihood BMAP_LinRegr')
axs[1].set_xlabel('Lambda Hyperparameter')
axs[1].set_ylabel('Out-of-sample LLHD')
axs[1].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
########now select best hyperparameter 





