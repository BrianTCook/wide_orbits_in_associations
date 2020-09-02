#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 17:14:16 2020

@author: BrianTCook
"""

import numpy as np
import glob
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

wide_orbits_direc = '/Users/BrianTCook/Desktop/wide_orbits_in_associations/'
energy_direc = wide_orbits_direc + 'data/LCC_energy_conservation/'

time_files = glob.glob(energy_direc + 'energy_times_*.txt')
energy_files = glob.glob(energy_direc + 'delta_energies_*.txt')

ratios = [ 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 ]

plt.figure(figsize=(5,5))

for dt_ratio, time_file, energy_file in zip(ratios, time_files, energy_files):
    
    print(dt_ratio)
    print(time_file)
    print(energy_file)
    print('')
    
    time, energy = np.loadtxt(time_file), np.abs(np.loadtxt(energy_file))
    time, energy = time[1:len(time)-1] , energy[1:len(time)-1]
    
    plt.semilogy(time, energy, label=r'$\log(\Delta t / t_{\rm dyn}) = %.01f$'%(np.log10(dt_ratio)))
    
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10)
plt.xlabel(r'$t_{\rm sim} [{\rm Myr}]$', fontsize=16) 
plt.ylabel(r'$|\Delta E|/E(t=0)$', fontsize=16) 