#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 08:56:49 2020

@author: BrianTCook
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_in_time/MCMC_*.txt')
times = np.linspace(0., 64., 9)

for i in [0, 1, 2]:
    
    plt.figure()
    
    if i == 0:
        
        plt_str = 'rho_naught'
        plt.xlabel(r'$\rho_{0} \hspace{4mm} [M_{\odot}/{\rm pc}^{3}]$', fontsize=16)
        
    if i == 1:
        
        plt_str = 'a'
        plt.xlabel(r'$a \hspace{4mm} [{\rm pc}]$', fontsize=16)
        
    if i == 2:

        plt_str = 'gamma'
        plt.xlabel(r'$\gamma$', fontsize=16)
    
    for j, data_set in enumerate(data):
        
        estimates = np.loadtxt(data_set)
        
        plt.hist(estimates[:,i], bins=50, density=True, histtype='step', color='C' + str(j), label=r'$t = %.0f {\rm Myr}$'%(times[j]))
        plt.gca().set_yscale('log')
    
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('MCMC_plot_%s.pdf'%(plt_str))
    plt.close()