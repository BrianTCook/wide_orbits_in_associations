#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:42:07 2020

@author: BrianTCook
"""

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import numpy as np
import glob

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

bg_str = 'with_background'
files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/StellarEvolution_*_LCC.ascii'%(bg_str))

sun_temp = 5778.
times = np.linspace(0., 64., 9)

for i, file in enumerate(files):

    stellar_data = np.loadtxt(file)
    masses, lums, temps = stellar_data[:,0], stellar_data[:,1], stellar_data[:,2]

    print('number of stars: %i'%(len(masses)))
    print('total mass: %.06f'%(np.sum(masses)))

    fig, ax = plt.subplots(figsize=[5, 4])
    
    scat = ax.scatter(temps, lums, c=np.log10(masses), 
                      s=1, marker=',', cmap='viridis')
        
    # inset axes....
    axins = ax.inset_axes([0.1, 0.1, 0.3, 0.3])
    
    axins.scatter(temps, lums, c=np.log10(masses), 
                      s=1, marker=',', cmap='viridis')
    
    x_med, y_med = sun_temp, 1.
    x1, x2, y1, y2 = x_med - 400., x_med + 400., y_med - 0.5, y_med + 0.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    
    axins.invert_xaxis()
    mark_inset(ax, axins, loc1=1, loc2=2, 
               joinstyle='bevel', fc="none", ec="0.0")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$T_{\rm eff} \hspace{2mm} [{\rm K}]$', fontsize=16)
    ax.set_ylabel(r'$L \hspace{2mm} [L_{\odot}]$', fontsize=16)
    ax.set_title(r'HR Diagram, $t = %.0f$ Myr'%(times[i]), fontsize=16)
    ax.set_xlim(1e3, 3.16e5)
    ax.set_ylim(1e-5, 1e5)
    cbar = fig.colorbar(scat)
    cbar.set_label(r'$\log_{10}(M_{\star} \hspace{1mm} [M_{\odot}])$', rotation=270, labelpad=20)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('HR_diagram_t_%i_Myr.pdf'%(times[i]))
    plt.close()