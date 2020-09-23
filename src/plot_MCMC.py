#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 08:56:49 2020

@author: BrianTCook
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

from MCMC_EFF_model import EFF_model

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

times = np.linspace(0., 64., 9)

for bg_str in ['with_background', 'without_background']:

    data = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/MCMC_*.txt'%(bg_str))
        
    for j, data_set in enumerate(data):
        
        print(j)
        
        estimates = np.loadtxt(data_set)
        
        if bg_str == 'without_background':
            
            theta = np.median(estimates[:, 0]), np.median(estimates[:, 1]), np.median(estimates[:, 2])
        
            rvals = np.linspace(0., 1.1*theta[1], 100)
            rho_vals = [ EFF_model(theta, r) for r in rvals ]
        
            plt.plot(rvals, rho_vals, color='C' + str(j),
                     label=r'$t = %.0f {\rm Myr}$'%(times[j]), linewidth=0.5)
            
            plt.gca().set_yscale('log')
    
    plt.legend(loc='lower right', fontsize=6)
    plt.tight_layout()
    plt.savefig('EFF_density_MCMC_estimates.pdf')
    plt.close()