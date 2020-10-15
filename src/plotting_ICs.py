#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 08:32:15 2020

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

for bg_str in background_strs:
    
    files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_LCC.ascii'%(bg_str))
    
    rho_0_present, a_present, gamma_present = 0.017964432528751385, 50.1, 15.2
    theta_present = rho_0_present, a_present, gamma_present

    for k, file in enumerate(files):

        print('-----------------------------')
        print(bg_str)
        print('time is: %.02f Myr'%(times[k]))
        
        data_init = np.loadtxt(file)
        
        mass_association = np.sum(data_init[:,0])
        
        print(data_init[:,0])
        
        center_of_mass = np.average(data_init[:,1:4], axis=0, weights=data_init[:,0])
        print(center_of_mass)
        
        positions = data_init[:,1:4]
        rvals = [ np.linalg.norm(position - center_of_mass) for position in positions ]
        
        df_init = pd.DataFrame(data=data_init, columns=('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        df_init.insert(1, 'Distance from COM', rvals, True)
        df_init = df_init.sort_values(by=['Distance from COM'])
        
        df_solar = df_init[df_init['mass'] > 0.8]
        df_solar = df_solar[df_solar['mass'] < 1.3]
        
        print('number of G type stars: %i'%(len(df_solar.index)))

        xvals_all = [ float(x)-center_of_mass[0] for x in df_init['x'].tolist() ]
        yvals_all = [ float(y)-center_of_mass[1] for y in df_init['y'].tolist() ]

        xvals_solar = [ float(x)-center_of_mass[0] for x in df_solar['x'].tolist() ]
        yvals_solar = [ float(y)-center_of_mass[1] for y in df_solar['y'].tolist() ]
        
        plt.figure(figsize=(6,6))
        plt.gca().set_aspect('equal')
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.plot(xvals_all, yvals_all, color='black',marker=',',lw=0, linestyle='', label='LCC members')
        plt.plot(xvals_solar, yvals_solar, color='red',marker='*',lw=0, linestyle='', label='G-type stars')
        plt.annotate(r'$M_{\rm LCC} = %.03f \hspace{2mm} M_{\odot}$'%(mass_association), xy=(0.6, 0.1), xycoords='axes fraction')
        plt.annotate(r'$\rho(r) \sim (1 + (r/a)^{2})^{-\gamma/2}$', xy=(0.6, 0.05), xycoords='axes fraction')
        plt.xlabel(r'$x-{x}_{\rm COM}$ (pc)', fontsize=16)
        plt.ylabel(r'$y-{y}_{\rm COM}$ (pc)', fontsize=16)
        plt.title(r'Lower Centaurus Crux model, $t=%.01f \, {\rm Myr}$'%(times[k]), fontsize=16)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('LCC_t_%.0f_Myr.pdf'%(times[k]))