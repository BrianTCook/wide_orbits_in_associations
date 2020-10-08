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
from scipy.spatial.distance import pdist, squareform
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

for j, t in enumerate(times):
    
    plt.figure()
    
    for bg_str in [ 'with_background' ]:

        data_phase = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_%s_*.ascii'%(bg_str, bg_str))
        data_MCMC = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/MCMC_*_%s.txt'%(bg_str, bg_str))

        model_parameters = np.loadtxt(data_MCMC[j])
        mass_and_phasespace = np.loadtxt(data_phase[j])
        
        rho_0_mode = mode_finder(model_parameters[:,0])
        a_mode = mode_finder(model_parameters[:,1])
        gamma_mode = mode_finder(model_parameters[:,2])
        
        theta = rho_0_mode, a_mode, gamma_mode
    
        center_of_mass = np.average(mass_and_phasespace[:,1:4], axis=0, weights=mass_and_phasespace[:,0])
        print(center_of_mass)
        
        positions = mass_and_phasespace[:,1:4]
        masses = mass_and_phasespace[:, 0]
        
        N = len(masses)
            
        rvals = np.linspace(0., 1.25*theta[1], 100)
        mass_EFF = [ enclosed_mass(theta, r) for r in rvals ]

        rvals_LCC = [ np.linalg.norm(position - center_of_mass) for position in positions ]
        
        df_init = pd.DataFrame(data=mass_and_phasespace, columns=('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        df_init.insert(1, 'Distance from COM', rvals_LCC, True)
        df_init = df_init.sort_values(by=['Distance from COM'])
            
        rvals_data = np.linspace(0., 1.25*theta[1], 40)
        mass_data = [ 0. for i in range(len(rvals_data)) ]
        
        delta_r = rvals_data[1] - rvals_data[0]
        shell_volumes = [ 4*np.pi*r**2. * delta_r for r in rvals_data ]
        
        for k, (r, shell) in enumerate(zip(rvals_data, shell_volumes)):
        
            #df = df_init[df_init['Distance from COM'] > r - delta_r/2.]
            #df = df[df['Distance from COM'] < r + delta_r/2.]
            
            df = df_init[df_init['Distance from COM'] < r]
            
            #rho_data[k] = np.sum(df['mass'].tolist()) / shell
            mass_data[k] = np.sum(df['mass'].tolist())
        
        if bg_str == 'with_background':

            plt.plot(rvals, mass_EFF, color='C' + str(j), linestyle='-.',
                     label=r'$t = %.0f \, \,  {\rm Myr, \, EFF \, model}$'%(times[j]), linewidth=0.5)
            plt.scatter(rvals_data, mass_data, color='C' + str(j), s=4,
                        label=r'$t = %.0f \, \,  {\rm Myr, \, data}$'%(times[j]))

    plt.xlabel(r'$r \, [{\rm pc}]$', fontsize=16)
    plt.ylabel(r'$M_{\rm enc}(<r) \, [M_{\odot}]$', fontsize=16)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('EFF_to_data_t_%i_Myr.pdf'%(t))
    plt.close()