#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:49:29 2023

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
    
    #compute cost of bmap for sible hyperparameter
    def compute_cost(self):
        BMAP, _ = self.run_BMAP_SGD() #get SGD BMAP
        m = len(self.y)
        predictions = self.X.dot(BMAP)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - self.y))
        return cost
    







##test the dunction see if it works
np.random.seed(0)  # for reproducibility

# Generating random dataset of size 1024x24 for X
Xdata = np.random.rand(1024, 24)
 #normalize data, suggestion online
Xdata_in = (Xdata - np.mean(Xdata, axis=0)) / np.std(Xdata, axis=0)

# Generating random continuous y values
y_data_in = np.random.rand(1024)

# Initialize and test the SGD_BMAP class
model = SGD_BMAP(Xdata=Xdata_in, y_data=y_data_in, alpha=0.01, numBatches=10)
beta_final, beta_history = model.run_BMAP_SGD()

print("Final Beta Values:", beta_final)











