#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:51:05 2020

@author: BrianTCook
"""

import matplotlib.pyplot as plt
import numpy as np
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

background_strs = [ 'with_background' ]
times = np.linspace(0., 64., 9)

bins_LCC = np.logspace(-2., 1., 40)
bins_planets = np.logspace(-3., -1., 40)

first_pop = np.loadtxt('first_population_planet_masses.txt')
first_pop_flattened = first_pop.flatten()

second_pop = np.loadtxt('second_population_planet_masses.txt')
third_pop = np.loadtxt('third_population_planet_masses.txt')


for bg_str in background_strs:
    
    files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_LCC.ascii'%(bg_str))
    
    rho_0_present, a_present, gamma_present = 0.017964432528751385, 50.1, 15.2
    theta_present = rho_0_present, a_present, gamma_present

    for k, file in enumerate(files):

        plt.figure()
        print('-----------------------------')
        print(bg_str)
        print('time is: %.02f Myr'%(times[k]))
        
        data_init = np.loadtxt(file)
        
        mass_association = np.sum(data_init[:,0])
        
        planets = np.concatenate([first_pop_flattened, second_pop, third_pop], axis=0)
        
        planets = [ p * 9.55e-4 for p in planets]
        
        plt.hist(data_init[:,0], bins=bins_LCC, histtype='step', label='LCC members')
        plt.hist(planets, bins=bins_planets, histtype='step', label='Planets')
        plt.axvline(x=11*9.55e-4, linewidth=0.5, label='HD 106906 b', color='k', linestyle='-.')
        plt.gca().set_xscale('log')
        #plt.gca().set_yscale('log')
        plt.xlabel('Mass \, [$M_{\odot}$]', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.legend(fontsize=8, loc='best')
        plt.tight_layout()
        plt.savefig('planet_and_star_mass_function_t_%.0f_Myr.pdf'%(times[k]))
        plt.close()