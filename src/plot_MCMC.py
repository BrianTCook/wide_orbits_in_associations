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
from scipy.special import hyp2f1

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def EFF_model(theta, r):
    
    rho_0, a, gamma = theta
    
    return rho_0 * ( 1 + (r/a)**2. )**(-gamma/2.)

def Jacobi_radius(theta, Mstar, rstar):
    
    rho_0, a, gamma = theta
    
    hyp2f1(3/2., (gamma+1.)/2., 5/2., -(rstar/a)**2.)
    
    return ((3*Mstar)/(4*np.pi*rho_0*hyp2f1(3/2., (gamma+1.)/2., 5/2., -(rstar/a)**2.)))**(1/3.)

def mode_finder(parameters):
    
    p_counts, p_edges = np.histogram(parameters, bins=100)
    p_index = np.where(p_counts == np.amax(p_counts))[0]
    p_mode = 0.5*(p_edges[p_index]+p_edges[p_index+1])[0]
    
    return p_mode

times = np.linspace(0., 64., 9)
bins = np.logspace(4.5, 6.5, 100)

for j, t in enumerate(times):
    
    plt.figure()
    
    for bg_str in ['with_background', 'without_background']:

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
        
        rvals = [ np.linalg.norm(position - center_of_mass) for position in positions ]
        
        Jacobi_radii = [ Jacobi_radius(theta, masses[i], rvals[i])*206265. for i in range(N) ] #AU
        print(min(Jacobi_radii))
    
        rvals = np.linspace(0., 1.25*theta[1], 100)
        rho_vals = [ EFF_model(theta, r) for r in rvals ]
    
        if bg_str == 'without_background':
            
            plt.hist(Jacobi_radii, bins=bins, color='b', histtype='step', linewidth=0.5,
                     label=r'$t = %.0f \, \,  {\rm Myr, \, no \, MW}$'%(times[j]))
            #plt.plot(rvals, rho_vals, color='C' + str(j), linestyle='-',
            #         label=r'$t = %.0f \, \, {\rm Myr, \, no \, MW}$'%(times[j]), linewidth=0.5)
            
        if bg_str == 'with_background':
            
            plt.hist(Jacobi_radii, bins=bins, color='r', histtype='step', linewidth=0.5,
                     label=r'$t = %.0f \, \,  {\rm Myr, \, MW}$'%(times[j]))
            #plt.plot(rvals, rho_vals, color='C' + str(j), linestyle='-.',
            #         label=r'$t = %.0f \, \,  {\rm Myr, \, MW}$'%(times[j]), linewidth=0.5)

        
    #plt.xlabel(r'$r \hspace{2mm} [{\rm pc}]$', fontsize=16)
    #plt.ylabel(r'$\rho(r) \hspace{2mm} [M_{\odot}/{\rm pc}^{3}]$', fontsize=16)
    plt.axvline(x=1.301*206265, linewidth=0.5, linestyle='-.', c='k', label=r'$r_{\rm sep}({\rm Sun}, \, {\rm Proxima \, \, Centauri})$')
    plt.xlabel(r'$r_{\rm J} \, \, [{\rm AU}]$', fontsize=16)
    plt.ylabel(r'Count', fontsize=16)
    plt.legend(loc='upper right', fontsize=8)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig('Jacobi_radii_t_%i_Myr.pdf'%(t))
    plt.close()