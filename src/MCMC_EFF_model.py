#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:40:56 2020

@author: BrianTCook
"""

from scipy.optimize import minimize
import numpy as np
import emcee

np.random.seed(42)

def EFF_model(theta, r):
    
    rho_0, a, gamma = theta
    
    return rho_0 * ( 1 + (r/a)**2. )**(-gamma/2.)

def log_prior(theta):
    
    rho_0, a, gamma = theta
    
    if rho_0 > 0. and a > 0. and gamma > 0.:
        
        return 0.
    
    return -np.inf

def log_likelihood(theta, r, rho, rho_err):
    
    rho_0, a, gamma = theta
    model = EFF_model(theta, r)
    
    sigma2 = rho_err ** 2
    
    return -0.5 * np.sum((rho - model) ** 2 / sigma2 + np.log(sigma2))

def log_probability(theta, r, rho, rho_err):
    
    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        
        return -np.inf
    
    return lp + (theta, r, rho, rho_err)

nll = lambda *args: -log_likelihood(*args)

mass_association = 700., r_max = 63. #from present epoch values

a_true, gamma_true = 50.1, 15.2 #from present epoch values
rho_0_true = 3 * mass_association / ( 4 * np.pi * r_max**3. * hyp2f1(3/2., (gamma+1.)/2., 5/2., -(r_max/a)**2.)) #solar masses per parsec

initial = np.array([rho_0_true, a_true, gamma_true]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(r, rho, rho_err))
rho_0_ml, a_ml, gamma_ml = soln.x

pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r, rho, rho_err))
sampler.run_mcmc(pos, 5000, progress=True)