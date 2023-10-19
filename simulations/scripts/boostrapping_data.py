#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:07:20 2023

@author: ahunos
"""

#define function to compute n batches
def createBatches(self):
    """
    Returns
    -------
    dict
        dictionary of X,y slices.
    """
    bSize = len(self.X) / nB # what shld each batch look lke
    
    #create indexes of data to use for gradient descent
    dataInd = [np.random.choice(range(len(self.X)), size = bSize, replace = None) for i in range(bSize)] 
    
    # fucntion slice data
    def make_data_slices(x, y, idx):
        X_data_slice = x[idx]
        y_data_slice = y[idx]
        return {"X":X_data_slice, "Y":y_data_slice}

#print(f'create {nB} slices of size {bSize} as dict of np.arrays')
#data_slices_dict = {f'slice{i}':make_data_slices(x = self.X, y= self.y, idx = vals) for i,vals in enumerate(dataInd)}
return X_data_slice