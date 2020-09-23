#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 08:56:49 2020

@author: BrianTCook
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

from scipy.spatial.distance import pdist, squareform

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def EFF_model(theta, r):
    
    rho_0, a, gamma = theta
    
    return rho_0 * ( 1 + (r/a)**2. )**(-gamma/2.)

times = np.linspace(0., 64., 9)
bins=np.logspace(np.log10(1e4),np.log10(1e8), 200)

for j, t in enumerate(times):
    
    plt.figure()
    
    for bg_str in ['with_background', 'without_background']:

        data = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_%s_*.ascii'%(bg_str, bg_str))
        data_set = data[j]

        estimates = np.loadtxt(data_set)
        theta = np.median(estimates[:, 0]), np.median(estimates[:, 1]), np.median(estimates[:, 2])
    
        rvals = np.linspace(0., 1.5*theta[1], 100)
        rho_vals = [ EFF_model(theta, r) for r in rvals ]
    
        if bg_str == 'without_background':
            
            plt.plot(rvals, rho_vals, color='C' + str(j), linestyle='-',
                     label=r'$t = %.0f \, \, {\rm Myr, \, no \, MW}$'%(times[j]), linewidth=0.5)
            
        if bg_str == 'with_background':
            
            plt.plot(rvals, rho_vals, color='C' + str(j), linestyle='-.',
                     label=r'$t = %.0f \, \,  {\rm Myr, \, MW}$'%(times[j]), linewidth=0.5)
        
    #plt.axvline(x=206265.*1.301, label=r'$r_{\rm sep}(\rm{Sun}, \, \rm{Proxima \, Centauri})$', linestyle='--', c='k', linewidth=0.5)
    #plt.xlabel(r'$r_{\rm sep} \, \, [{\rm AU}]$', fontsize=16)
    #plt.ylabel(r'Count', fontsize=16)
    plt.legend(loc='lower left', fontsize=10)
    #plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig('EFF_models_t_%i_Myr.pdf'%(t))
    plt.close()