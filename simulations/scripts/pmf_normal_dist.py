#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:07:53 2023

@author: ahunos
"""

#siulation of normal distribution
#pmf of normal
#P(X | \theta, sigma_2) 
import numpy as np

#define the likelihood of normal distribtion
def pmf_norm(y, sigma, mu):
    exp_term = np.exp((-(1/2) * ((y - mu) / sigma)**2))
    first_term=(1 / np.sqrt(2 * np.pi* sigma**2))
    return np.sum(first_term * exp_term)
    