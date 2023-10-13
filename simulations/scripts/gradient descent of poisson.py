# gradient descent of poisson 

# steps
#1. function to calculate the gradient (wrt lambda) of the poisson log likelihood
#2. function to implement gradient descent

import numpy as np
def pois_grad(X, lambda_val):
    return np.sum(-1 + (X)/lambda_val)

    
def grad_desc_pois(X_data, lambda_init=0.1, max_iter = 100, stepSize=0.01):
    #set init lammda 
    lambda_value=lambda_init
    for iter in range(max_iter):
        #compute gradient
        gradient_cal = pois_grad(X=X_data, lambda_val=lambda_value)
        #update lambda
        lambda_value += stepSize*gradient_cal
        # lambda_new = lambda_value
        # print(f'iter {iter}; lambda old{lambda_value}')
        #this is to avoid negative lambda values
        max(0.001, lambda_value)
        
    return lambda_value



#now run func 
data=np.random.poisson(3, 10)
grad_desc_pois(X_data=data, lambda_init=0.1, max_iter = 100, stepSize=0.01)


##########now implement to converge

def opt_grad_desc_pois(X_data, lambda_init=0.1, max_iter = 100, stepSize=0.01, tol_val = 0.00001):
    #set up
    #use a while loop, while iteration < max_iter or abs(lambda_new - lamnda_pre) > tol_val, run while loop
    #break if abs(lambda_new - lamnda_pre) <= tol
    
    num_iter = 0 #set up iteration
    lambda_ = lambda_init
    #this approach means we have a prior knowledge of where lambda value should be
    lambda_prev = lambda_init + 10 * tol_val  # Initialize with a difference to enter the loop
    print(f'translate to i think lambda be around {lambda_prev}')

    while np.abs(lambda_ - lambda_prev) > tol_val and num_iter < max_iter:
        grad_val = pois_grad(X=X_data, lambda_val=lambda_) #cal gradient wrt lambda
        lambda_prev = lambda_ #set lambda to previous lambda
        lambda_ += stepSize * grad_val #update the gradient
        num_iter += 1
    if np.abs(lambda_ - lambda_prev) <= tol_val:
        print(f'converged at {num_iter} for lambda_val {lambda_}')
    else:
        print(f'may have not converged after {max_iter} iterations')
    return lambda_

opt_grad_desc_pois(X_data=data, lambda_init=0.1, max_iter = 100, stepSize=0.01, tol_val = 0.00001)
