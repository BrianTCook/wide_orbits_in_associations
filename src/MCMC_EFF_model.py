#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:40:56 2020

@author: BrianTCook
"""

from scipy.optimize import minimize
from scipy.special import hyp2f1
import numpy as np
import pandas as pd
import emcee
import glob
import matplotlib.pyplot as plt
import corner

np.random.seed(42)

def EFF_model(theta, r):
    
    rho_0, a, gamma = theta
    
    return rho_0 * ( 1 + (r/a)**2. )**(-gamma/2.)

def log_prior(theta):
    
    rho_0, a, gamma = theta
    
    if rho_0 > 0. and a > 0. and gamma > 0. and a < 200. and gamma < 30.:
        
        return 0.
    
    return -np.inf

def log_likelihood(theta, r, rho, rho_err):
    
    rho_0, a, gamma = theta
    N = len(r)
    
    model = [ EFF_model(theta, rval) for rval in r ]
    sigma2 = [ err**2. for err in rho_err ]
    
    summand = [ (rho[i] - model[i]) ** 2. / sigma2[i] + np.log(sigma2[i]) for i in range(N) ]

    try:
    
        return -0.5 * np.sum(summand)
    
    except:
        
        return -np.inf

def log_probability(theta, r, rho, rho_err):
    
    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        
        return -np.inf
    
    return lp + log_likelihood(theta, r, rho, rho_err)

nll = lambda *args: -log_likelihood(*args)

files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_in_time/phasespace_*_LCC.ascii')
times = np.linspace(0., 64., 9)

for i, file in enumerate(files):

    print('time is: %.02f Myr'%(times[i]))
    
    data_init = np.loadtxt(file)
    
    center_of_mass = np.average(data_init[:,1:4], axis=0, weights=data_init[:,0])
    print(center_of_mass)
    
    positions = data_init[:,1:4]
    rvals = [ np.linalg.norm(position - center_of_mass) for position in positions ]
    
    df_init = pd.DataFrame(data=data_init, columns=('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
    df_init.insert(1, 'Distance from COM', rvals, True)
    df_init = df_init.sort_values(by=['Distance from COM'])
    
    mass_association = np.sum(data_init[:,0]) #from present epoch values
    r_max = 63.945 #from present epoch values
    a_true, gamma_true = 50.1, 15.2 #from present epoch values
    rho_0_true = 3 * mass_association / ( 4 * np.pi * r_max**3. * hyp2f1(3/2., (gamma_true+1.)/2., 5/2., -(r_max/a_true)**2.)) #solar masses per parsec
    
    Nbins = 50
    r_edges = np.linspace(0., r_max, Nbins+1) #edges
    r_centers = [ 0.5*(r_edges[i]+r_edges[i+1]) for i in range(Nbins) ] #centers
    delta_r = r_centers[1] - r_centers[0]
    
    shell_volumes = [ 4*np.pi*r**2. * delta_r for r in r_centers ]
    
    rho = [ 0. for i in range(Nbins) ]
    
    for j, (r, shell) in enumerate(zip(r_centers, shell_volumes)):
    
        df = df_init[df_init['Distance from COM'] > r - delta_r/2.]
        df = df[df['Distance from COM'] < r + delta_r/2.]
        
        rho[j] = np.sum(df['mass'].tolist()) / shell
    
    rho_err = [ 0.1*rhoval for rhoval in rho ]
    
    r = r_centers
    
    initial = np.array([rho_0_true, a_true, gamma_true]) + 0.1 * np.random.randn(3)
    soln = minimize(nll, initial, args=(r, rho, rho_err))
    rho_0_ml, a_ml, gamma_ml = soln.x
    
    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r, rho, rho_err))
    sampler.run_mcmc(pos, 20000, progress=True)

    '''
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\rho_{0}$", r"$a$", r"$gamma$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number")
    '''
    
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True) 
    np.savetxt('MCMC_parameters_%i.txt'%(i), flat_samples)
    fig = corner.corner(flat_samples, labels=labels, truths=[rho_0_true, a_true, gamma_true])