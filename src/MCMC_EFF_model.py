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

def gaussian(x, mu, sig):
    
    return np.exp( -(x - mu)**(2.) / (2. * sig**(2.)) )

def EFF_model(theta, r):
    
    rho_0, a, gamma = theta
    
    return rho_0 * ( 1 + (r/a)**2. )**(-gamma/2.)

def enclosed_mass(theta, r):
    
    rho_0, a, gamma = theta
    mass_enc = (4*np.pi / 3.) * rho_0 * r**(3.) * hyp2f1(3/2., (gamma+1.)/2., 5/2., -(r/a)**2.) #solar masses

    return mass_enc #no units, although we will need in terms of MSun

def sigma_rho(theta_present, r, delta_r):
    
    rho_0_present, a_present, gamma_present = theta_present
    
    return -gamma_present * (r/(a_present)**(2.)) * (1 + (r/a_present)**(2.))**(-gamma_present/2. - 1) * (delta_r / 10.)

def r_max_finder(mass_association, a, gamma):
    
	#normalizing factor has to be 
	min_star_mass = 0.02

	delta_r = 2. #width of bins in parsecs

	matching_value = 1/3. * (a/delta_r) * min_star_mass / mass_association

	def f(u):

		return 1./u * (1+u**(2.))**(-gamma/2.) - matching_value

	u_min, u_max = 1e-3, 5.
	delta = u_max - u_min

	while True:

		if f(u_min + delta/2.) * f(u_max) < 0.:

			u_min += delta/2.

		if f(u_min) * f(u_max - delta/2.) < 0.:

			u_max -= delta/2.

		delta = u_max - u_min

		if delta < 1e-6:

			return 0.5*(u_min + u_max) * a

def log_prior(theta):
    
    rho_0, a, gamma = theta
    if rho_0 > 0. and a > 0. and gamma > 0.:
        return 0.0
    return -np.inf

def log_likelihood(theta, r, mass_enc, mass_enc_err):
    
    rho_0, a, gamma = theta
    N = len(r)
    
    model = [ enclosed_mass(theta, rval) for rval in r ]
    
    summand = [ ((mass_enc[i] - model[i])/(mass_enc_err[i]))**(2.) + np.log(2*np.pi*mass_enc_err[i]**(2.)) for i in range(N) ]
    
    if np.isnan(np.sum(summand)):
        return -np.inf
    return -0.5*np.sum(summand)

def log_probability(theta, r, mass_enc, mass_enc_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):        
        return -np.inf    
    return lp + log_likelihood(theta, r, mass_enc, mass_enc_err)

nll = lambda *args: -log_likelihood(*args)
times = np.linspace(0., 64., 9)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

background_strs = [ 'with_background', 'without_background' ]

for bg_str in background_strs:
    
    files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_LCC.ascii'%(bg_str))
    
    rho_0_present, a_present, gamma_present = 0.017964432528751385, 50.1, 15.2
    theta_present = rho_0_present, a_present, gamma_present

    data_init = np.loadtxt(files[0])
    mass_association = np.sum(data_init[:,0])
    r_max = r_max_finder(mass_association, a_present, gamma_present)

    for k, file in enumerate(files):

        print('-----------------------------')
        print(bg_str)
        print('time is: %.02f Myr'%(times[k]))
        
        data_init = np.loadtxt(file)
        
        center_of_mass = np.average(data_init[:,1:4], axis=0, weights=data_init[:,0])
        print(center_of_mass)
        
        positions = data_init[:,1:4]
        rvals = [ np.linalg.norm(position - center_of_mass) for position in positions ]
        
        df_init = pd.DataFrame(data=data_init, columns=('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        df_init.insert(1, 'Distance from COM', rvals, True)
        df_init = df_init.sort_values(by=['Distance from COM'])
        
        Nbins = 10
        r_edges = np.linspace(1., r_max, Nbins+1) #edges
        r = [ 0.5*(r_edges[i]+r_edges[i+1]) for i in range(Nbins) ] #centers
        
        mass_enc = [ 0. for i in range(Nbins) ]
        
        for j, rval in enumerate(r):
        
            df = df_init[df_init['Distance from COM'] < rval]
            mass_enc[j] = np.sum(df['mass'].tolist())
        
        print(mass_enc)
        
        mass_enc_err = [ 0.05*mass_enc[i] for i in range(Nbins) ] 
        
        nll = lambda *args: -log_likelihood(*args)
        initial = np.array([rho_0_present, a_present, gamma_present]) + 0.1 * np.random.randn(3)
        soln = minimize(nll, initial, args=(r, mass_enc, mass_enc_err))
        rho_0_ml, a_ml, gamma_ml = soln.x
        
        #r_max = a_ml
        
        print("Maximum likelihood estimates:")
        print("rho_0 = {0:.3f}".format(rho_0_ml))
        print("a = {0:.3f}".format(a_ml))
        print("gamma = {0:.3f}".format(gamma_ml))
        
        pos = soln.x + 1e-4 * np.random.randn(64, 3)
        nwalkers, ndim = pos.shape
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r, mass_enc, mass_enc_err))
        sampler.run_mcmc(pos, 60000, progress=True)
        
        flat_samples = sampler.get_chain(discard=10000, thin=5, flat=True) 
        
        np.savetxt('MCMC_data_%i_Myr_%s.txt'%(times[k], bg_str), flat_samples)
        
        labels = [ r'$\rho_{0} \hspace{2mm} [M_{\odot} \, {\rm pc}^{-3}]$', r'$a \hspace{2mm} [{\rm pc}]$', r'$\gamma$' ]
        fig = corner.corner(flat_samples, labels=labels, truths=[rho_0_present, a_present, gamma_present])
        plt.suptitle(r'$\rho(r, t = %.0f \hspace{2mm} {\rm Myr}) \simeq \rho_{0}\left(1+(r/a)^{2}\right)^{-\gamma/2}$'%(times[k]), fontsize=18, x=0.6666, y=0.83333)
        plt.subplots_adjust(top=0.96)
        plt.savefig('MCMC_histograms_time_%s_Myr_%s.pdf'%(str(int(times[k])), bg_str))
        plt.close()
        print('-----------------------------')