#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:08:33 2020

@author: BrianTCook
"""

import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import glob

from scipy.special import hyp2f1
from scipy.spatial.distance import pdist, squareform

from galpy.potential import MWPotential2014
from galpy.util import bovy_conversion

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def mode_finder(parameters):
    
    p_counts, p_edges = np.histogram(parameters, bins=100)
    p_index = np.where(p_counts == np.amax(p_counts))[0]
    p_mode = 0.5*(p_edges[p_index]+p_edges[p_index+1])[0]
    
    return p_mode

def beta(star_data, theta, LCC_center_of_mass):
    
    m_star, x_star, y_star, z_star, vx_star, vy_star, vz_star = star_data
    rho_0, a, gamma = theta
    
    r_GC_star = np.sqrt(x_star**2. + y_star**2. + z_star**2.) #in parsecs
    r_LCC_star = np.linalg.norm([x_star, y_star, z_star] - LCC_center_of_mass) #parsecs
    
    Rgal, zgal = np.sqrt(x_star**2. + y_star**2.)/1000., z_star/1000. #in kpc

    Mgal = 0. #solar masses

    for pot in MWPotential2014:

        Mgal += pot.mass(Rgal, zgal) * bovy_conversion.mass_in_msol(220., 8.)
    
    ratio = (1./r_GC_star) * ((3*Mgal)/(4*np.pi*rho_0*hyp2f1(3/2., (gamma+1.)/2., 5/2., -(r_LCC_star/a)**2.)))**(1/3.)
    
    return np.log10(ratio)

times = np.linspace(0., 64., 9)

for j, t in enumerate(times):
    
    plt.figure()
    
    bg_str = 'with_background'

    data_phase = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/PhaseSpace_*_%s_*.ascii'%(bg_str, bg_str))
    data_MCMC = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/forward_%s/MCMC_*_%s.txt'%(bg_str, bg_str))

    model_parameters = np.loadtxt(data_MCMC[j])
    mass_and_phasespace = np.loadtxt(data_phase[j])

    N = len(mass_and_phasespace[:,0])
    
    LCC_center_of_mass = np.average(mass_and_phasespace[:,1:4], axis=0, weights=mass_and_phasespace[:,0])
    
    rho_0_mode = mode_finder(model_parameters[:,0])
    a_mode = mode_finder(model_parameters[:,1])
    gamma_mode = mode_finder(model_parameters[:,2])
    
    theta = rho_0_mode, a_mode, gamma_mode
    
    beta_vals = [ beta(mass_and_phasespace[k,:], theta, LCC_center_of_mass) for k in range(N) ]
    
    distances = squareform(pdist(mass_and_phasespace[:,1:4]))
    min_distance_vals = [ np.log10(206265. * np.min(distances[k,:][np.nonzero(distances[k,:])])) for k in range(N) ]
    
    h = sns.jointplot(x=beta_vals, y=min_distance_vals)

    # JointGrid has a convenience function
    h.set_axis_labels(r'$\beta \equiv \log_{10}\left(r_{\rm H, LCC}/r_{\rm H, MW}\right)$', r'$\log_{10} \left({\rm min}\left({\bf r}_{\rm sep}\right) \, [{\rm AU}]\right)$', fontsize=16)
    h.ax_marg_x.set_xlim(-1., 1.)
    h.ax_marg_y.set_ylim(2., 7.)
    
    h.fig.suptitle(r'$t_{\rm sim} = %.0f \, {\rm Myr}$'%(t))
    plt.tight_layout()
    plt.savefig('beta_vs_minsep_t_%.0f_Myr.pdf'%(t))