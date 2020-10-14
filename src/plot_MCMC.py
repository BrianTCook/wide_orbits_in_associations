#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 08:56:49 2020

@author: BrianTCook
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

import pandas as pd
from scipy.stats import chisquare, kstest, ks_2samp
from scipy.special import hyp2f1

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def EFF_model(theta, r):
    
    rho_0, a, gamma = theta
    
    return rho_0 * ( 1 + (r/a)**2. )**(-gamma/2.)

def enclosed_mass(theta, r):
    
    rho_0, a, gamma = theta
    mass_enc = (4*np.pi / 3.) * rho_0 * r**(3.) * hyp2f1(3/2., (gamma+1.)/2., 5/2., -(r/a)**2.) #solar masses

    return mass_enc #no units, although we will need in terms of MSun

def Jacobi_radius(theta, Mstar, rstar):
    
    rho_0, a, gamma = theta
    
    return ((3*Mstar)/(4*np.pi*rho_0*rstar**(3.)*hyp2f1(3/2., (gamma+1.)/2., 5/2., -(rstar/a)**2.)))**(1/3.)

def mode_finder(parameters):
    
    p_counts, p_edges = np.histogram(parameters, bins=100)
    p_index = np.where(p_counts == np.amax(p_counts))[0]
    p_mode = 0.5*(p_edges[p_index]+p_edges[p_index+1])[0]
    
    return p_mode

times = np.linspace(0., 64., 9)
bins = np.logspace(3., 6., 100)

rho_0_present, a_present, gamma_present = 0.017964432528751385, 50.1, 15.2

plt.figure()

rho_0_lower, rho_0_modes, rho_0_upper = [], [], []
a_lower, a_modes, a_upper = [], [], []
gamma_lower, gamma_modes, gamma_upper = [], [], []

for j, t in enumerate(times):
    
    for bg_str in [ 'with_background', 'without_background' ]:

        data_phase = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_%s_*.ascii'%(bg_str, bg_str))
        data_MCMC = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/MCMC_*_%s.txt'%(bg_str, bg_str))

        model_parameters = np.loadtxt(data_MCMC[j])
        mass_and_phasespace = np.loadtxt(data_phase[j])
        
        rho_0_values = model_parameters[:,0]
        a_values = model_parameters[:,1]
        gamma_values = model_parameters[:,2]
        
        rho_0_mode = mode_finder(rho_0_values)
        a_mode = mode_finder(a_values)
        gamma_mode = mode_finder(gamma_values)
        
        theta = rho_0_mode, a_mode, gamma_mode
    
        center_of_mass = np.average(mass_and_phasespace[:,1:4], axis=0, weights=mass_and_phasespace[:,0])
        print('t = %.0f Myr, %s'%(t, bg_str))
        
        positions = mass_and_phasespace[:,1:4]
        masses = mass_and_phasespace[:, 0]
        
        N = len(masses)
        print(np.sum(masses))
            
        rvals = np.linspace(0., 60., 500)        
        
        mass_EFF = [ enclosed_mass(theta, r) for r in rvals ]
        
        rvals_LCC = [ np.linalg.norm(position - center_of_mass) for position in positions ]
        
        df_init = pd.DataFrame(data=mass_and_phasespace, columns=('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        df_init.insert(1, 'Distance from COM', rvals_LCC, True)
        df_init = df_init.sort_values(by=['Distance from COM'])
            
        rvals_data = np.linspace(0., 60., 20)
        mass_data = [ 0. for i in range(len(rvals_data)) ]
        
        delta_r = rvals_data[1] - rvals_data[0]
        shell_volumes = [ 4*np.pi*r**2. * delta_r for r in rvals_data ]
        
        for k, (r, shell) in enumerate(zip(rvals_data, shell_volumes)):
        
            #df = df_init[df_init['Distance from COM'] > r - delta_r/2.]
            #df = df[df['Distance from COM'] < r + delta_r/2.]
            
            df = df_init[df_init['Distance from COM'] < r]
            
            #rho_data[k] = np.sum(df['mass'].tolist()) / shell
            mass_data[k] = np.sum(df['mass'].tolist())
        
        plt.plot(rvals, mass_EFF, color='C' + str(j), linewidth=0.5,
                 label=r'$t = %.0f \, \,  {\rm Myr, \, EFF \, model}$'%(times[j]))
        plt.scatter(rvals_data, mass_data, color='C' + str(j), s=4,
                    label=r'$t = %.0f \, \,  {\rm Myr, \, data}$'%(times[j]))
        
        statistic, pvalue = ks_2samp(np.asarray(mass_data), np.asarray(mass_EFF))
        
        plt.annotate(r'KS Test Statistic and $p$-value: %.03f, %.03f'%(statistic, pvalue), xy=(0.4, 0.1), xycoords='axes fraction')
        print('KS statistic and pvalue: ', statistic, pvalue)
        print('')
        
        plt.xlabel(r'$r \, [{\rm pc}]$', fontsize=16)
        plt.ylabel(r'$M_{\rm enc}(<r) \, [M_{\odot}]$', fontsize=16)
        plt.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig('EFF_to_data_t_%i_Myr_%s.pdf'%(t, bg_str))
        plt.show()
        plt.close()

'''
rho_0_values = model_parameters[:,0]
        a_values = model_parameters[:,1]
        gamma_values = model_parameters[:,2]
        
        rho_0_mode = mode_finder(rho_0_values)
        a_mode = mode_finder(a_values)
        gamma_mode = mode_finder(gamma_values)
        
        rho_0_lower.append(np.percentile(rho_0_values, (100-99.73))/rho_0_present)
        rho_0_modes.append(rho_0_mode/rho_0_present)
        rho_0_upper.append(np.percentile(rho_0_values, 99.73)/rho_0_present)
        
        a_lower.append(np.percentile(a_values, (100-99.73))/a_present)
        a_modes.append(a_mode/a_present)
        a_upper.append(np.percentile(a_values, 99.73)/a_present)
        
        gamma_lower.append(np.percentile(gamma_values, (100-99.73))/gamma_present)
        gamma_modes.append(gamma_mode/gamma_present)
        gamma_upper.append(np.percentile(gamma_values, 99.73)/gamma_present)

print(a_modes)

plt.scatter(times, rho_0_modes, color='C0', marker='.', s=8, label=r'core density $\rho_{0}$ ($\pm 3\sigma$)')
plt.vlines(times, rho_0_lower, rho_0_upper, color='C0')
plt.scatter(times, a_modes, color='C1', marker='.', s=8, label=r'scale radius $a$ ($\pm 3\sigma$)')
plt.vlines(times, a_lower, a_upper, color='C1')
plt.scatter(times, gamma_modes, color='C2', marker='.', s=8, label=r'power law index $\gamma$ ($\pm 3\sigma$)')
plt.vlines(times, gamma_lower, gamma_upper, color='C2')
plt.axhline(y=1, linewidth=0.5, linestyle='-.', color='k')

plt.xlabel(r'$t \, [{\rm Myr}]$', fontsize=16)
plt.ylabel(r'$\theta_{\rm MCMC}/\theta_{\rm obs, \, present}$', fontsize=16)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('MCMC_vs_observations.pdf')
'''