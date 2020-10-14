#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:51:05 2020

@author: BrianTCook
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

background_strs = [ 'with_background' ]
times = np.linspace(0., 64., 9)

bins = np.logspace(-4., 1., 50)
print(bins)

for bg_str in background_strs:
    
    files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_LCC.ascii'%(bg_str))
    
    rho_0_present, a_present, gamma_present = 0.017964432528751385, 50.1, 15.2
    theta_present = rho_0_present, a_present, gamma_present

    plt.figure()

    for k, file in enumerate(files):

        print('-----------------------------')
        print(bg_str)
        print('time is: %.02f Myr'%(times[k]))
        
        data_init = np.loadtxt(file)
        
        mass_association = np.sum(data_init[:,0])
        
        
        plt.hist(data_init[:,0], bins=bins, histtype='step', label='Stars')
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.legend(loc='best')
        plt.show()