#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:21:59 2020

@author: BrianTCook
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

np.random.seed(42)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

background_strs = [ 'with_background', 'without_background' ]
times = np.linspace(0., 64., 9)

def cross_section(r1, r2, v1, v2):
    
    r = r2 - r1
    v = v2 - v1
    
    b = np.linalg.norm(r) * (np.dot(r, v))/(np.linalg.norm(r)*np.linalg.norm(v)) * 206265. #AU
    
    if np.isnan(b):
    
        return 0. 
    
    return np.pi * b**(2.)

for bg_str in background_strs:
    
    files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_LCC.ascii'%(bg_str))
    
    rho_0_present, a_present, gamma_present = 0.017964432528751385, 50.1, 15.2
    theta = rho_0_present, a_present, gamma_present
    
    thetas = theta

    for k, file in enumerate(files):

        print('-----------------------------')
        print(bg_str)
        print('time is: %.02f Myr'%(times[k]))
        
        data_init = np.loadtxt(file)
        
        mass_association = np.sum(data_init[:,0])
        
        center_of_mass = np.average(data_init[:,1:4], axis=0, weights=data_init[:,0])
        print(center_of_mass)
        
        positions = data_init[:,1:4]
        rvals = [ np.linalg.norm(position - center_of_mass) for position in positions ]
        
        df_init = pd.DataFrame(data=data_init, columns=('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        df_init.insert(1, 'Distance from COM', rvals, True)
        df_init = df_init.sort_values(by=['Distance from COM'])
        
        df_array = df_init.to_numpy()
        
        N = len(df_init.index)
        
        print('gets here')
        cross_sections = [ cross_section(df_array[i, 2:5], df_array[j, 2:5], df_array[i, 5:8], df_array[j, 5:8]) 
                           for i in range(N) for j in range(i+1, N) ]

        print('here too')
        r_seps = [ np.linalg.norm(df_array[j, 2:5] - df_array[i, 2:5]) * 206265.
                   for i in range(N) for j in range(i+1, N) ]
        
        xbins = np.logspace(4., 8., 300)
        ybins = np.logspace(0., 18., 300)

        
        print('here three')
        # Small bins
        fig, ax = plt.subplots()
        h = ax.hist2d(r_seps, cross_sections, bins=[xbins, ybins], norm=LogNorm())
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$|\mathbf{r}_{j}-\mathbf{r}_{i}| \, [{\rm AU}]$', fontsize=14)
        ax.set_ylabel(r'$\sigma\left(\mathbf{w}_{i}, \mathbf{w}_{j}\right) \, [{\rm AU}^{2}]$', fontsize=14)
        fig.colorbar(h[3], ax=ax)
        plt.annotate(r'$t=%.0f \, {\rm Myr}$'%(times[k]), xy=(0.1, 0.8), xycoords='axes fraction')222
        plt.title('Stellar Interaction Cross Sections', fontsize=12)
        plt.tight_layout()
        plt.savefig('cross_section_and_separation_t_%.0f_Myr_%s.pdf'%(times[k], bg_str))