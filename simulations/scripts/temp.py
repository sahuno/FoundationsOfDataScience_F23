# -*- coding: utf-8 -*-
"""
Spyder Editor
Samuel ahuno f
Given: Y = np.array([3,6,2,9])
scipy.special.gammaln(Y + 1) is same as [np.log(np.math.factorial(i)) for i in Y].
"""

#import needed libraries
import numpy as np
from scipy.special import gammaln

Y = np.array([3,6,2,9])

np.math.factorial(Y) #this won't work, cos needs only scalar
gammaln(Y+1) #convenience function to cal log factorial of items in array


#demonstrate quality 
for i in Y:
    #print(i)
    print(np.log(np.math.factorial(i)))
#comprare to previous line from for loop
gammaln(Y+1) 