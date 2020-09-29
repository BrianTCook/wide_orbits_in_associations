#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:12:55 2020

@author: BrianTCook
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from amuse.lab import *

def find_nearest_index(value, array):

	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()

	return idx

def simpson(f, x_init, x_final, N_simp): #Simpson's integration rule, \int_{a}^{b} f(x) dx with N sample points

	h = (x_final-x_init)/N_simp

	I = f(x_init) + f(x_final)

	odds = [4*f(x_init + k*h) for k in range(1,N_simp,2)]
	evens = [2*f(x_init + k*h) for k in range(2,N_simp,2)]
	I += sum(odds) + sum(evens)

	I *= h/3.

	return I

def Cook_mass_fn(Nstars):

    '''
    piecewise mass function: for <= 1.4 MSun Goldman(2018), for > 1.4 MSun Kroupa(2001)
    Kroupa oversamples low-mass stars (especially with AMUSE), Goldman undersamples high-mass stars
    
    Uses the PDF and rejection sampling to return Nstars that follow this new mass function
    
    I made it up, so it gets named after me :)
    '''
    
    mean, sigma = 0.22, 0.64 #moving group data
    
    def log_normal_PDF(Mstar):
        
        return 1/(Mstar * sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(Mstar)-mean)**2./(2*sigma**2.))
        
    def kroupa_PDF(Mstar):
        
        #Mstar has to be in solar masses
        
        return log_normal_PDF(1.4)*(Mstar / 1.4)**(-2.3)
    
    def PDF(Mstar):
        
        if Mstar <= 1.4:
            
            return log_normal_PDF(Mstar)
        
        else:
            
            return kroupa_PDF(Mstar)
        
    def CDF(Mstar):
        
        if Mstar <= 1.4:
            
            return 0.5*(1 + erf((np.log(Mstar)-mean)/(np.sqrt(2)*sigma)))
        
        else:
        
            return CDF(1.4) + simpson(PDF, 1.4, Mstar, 100)
    
    log_min, log_max = np.log10(0.02), np.log10(17.5)
    delta_m = 0.05 #width of bins in MSun
    
    bin_edges = np.arange(10**(log_min), 10**(log_max), delta_m) #edges of bins in parsecs
    Nbins = len(bin_edges) - 1
    
    bin_centers = [ 0.5*(bin_edges[i] + bin_edges[i+1]) for i in range(Nbins) ]
    
    cdf = [ CDF(m) for m in bin_centers ]
    max_cdf = max(cdf)
    cdf_normalized = [ x/max_cdf for x in cdf ]
    
    allowances = [ 0. for i in range(Nbins) ]
    
    for i in range(Nbins):
    
    	allowances[i] = cdf_normalized[i] - np.sum(allowances[:i])	 #mass per slice allowed
        
    max_allowance = max(allowances)
        
    list_of_masses = []
        
    while len(list_of_masses) < Nstars:
        
        random_x = 10**((log_max - log_min)*np.random.uniform() + log_min)
        random_y = max_allowance * np.random.uniform()

        idx_bin = find_nearest_index(random_x, bin_centers)	
        
        if random_y < allowances[idx_bin]:
            
            new_mass = delta_m * np.random.uniform() + bin_edges[idx_bin]
            list_of_masses.append(new_mass)
            
    return list_of_masses
    
