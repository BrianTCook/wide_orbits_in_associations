#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:22:20 2020

@author: BrianTCook
"""

import numpy as np
import glob
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

wide_orbits_direc = '/Users/BrianTCook/Desktop/wide_orbits_in_associations/'

phasespace_files = glob.glob(wide_orbits_direc + 'data/phasespace_*.ascii')
times = np.loadtxt(wide_orbits_direc + 'data/snapshot_times.txt')

times = [ 15. + t for t in times ]

galaxy_mass_list = np.loadtxt(wide_orbits_direc + 'data/snapshot_galaxy_masses.txt')
masses = np.loadtxt(wide_orbits_direc + 'data/LCC_masses.txt')

total_mass = np.sum(masses)

hl_radius = 15.48
angles = np.linspace(0., 2*np.pi, 100)
xcirc = [ hl_radius * np.cos(angle) for angle in angles ]
ycirc = [ hl_radius * np.sin(angle) for angle in angles ]

for j, file in enumerate(phasespace_files):

    print('j: %i'%(j))
    
    data = np.loadtxt(file)

    Nobjects = len(data[:,0])
    xvals_stars_and_planets = data[:,0]
    yvals_stars_and_planets = data[:,1]
    zvals_stars_and_planets = data[:,2]
    
    x_med = np.median(xvals_stars_and_planets)
    y_med = np.median(yvals_stars_and_planets)
    z_med = np.median(zvals_stars_and_planets)
    
    R_GC = np.sqrt(x_med**2. + y_med**2. + z_med**2.)
        
    xvals_LCC_frame = [ x-x_med for x in xvals_stars_and_planets ]
    yvals_LCC_frame = [ y-y_med for y in yvals_stars_and_planets ]
    zvals_LCC_frame = [ z-z_med for z in zvals_stars_and_planets ]


    if j == 0:

        rvals_LCC_frame_naught = [ np.sqrt(xvals_LCC_frame[i]**2. + yvals_LCC_frame[i]**2.) for i in range(Nobjects) ]
        
        hist_naught, edges_naught = np.histogram(rvals_LCC_frame_naught, bins='auto')
        rvals_plot_naught = [ 0.5*(edges_naught[i]+edges_naught[i+1]) for i in range(len(edges_naught)-1) ]
        
        dr_naught = rvals_plot_naught[1] - rvals_plot_naught[0]
        annuli_areas = [ 2 * np.pi * r * dr_naught for r in rvals_plot_naught ]
        
        hist_naught = [ hist_naught[i]/annuli_areas[i] for i in range(len(annuli_areas)) ]
        
    if j > 0:
        
        rvals_LCC_frame = [ np.sqrt(xvals_LCC_frame[i]**2. + yvals_LCC_frame[i]**2.) for i in range(Nobjects) ]
        
        hist, edges = np.histogram(rvals_LCC_frame, bins='auto')
        rvals_plot = [ 0.5*(edges[i]+edges[i+1]) for i in range(len(edges)-1) ]
        
        dr = rvals_plot[1] - rvals_plot[0]
        annuli_areas = [ 2 * np.pi * r * dr for r in rvals_plot ]
        
        hist = [ hist[i]/annuli_areas[i] for i in range(len(annuli_areas)) ]
        
    fig, axs = plt.subplots(ncols=2)
    
    ax1 = axs[0]
    ax1.set_aspect('equal')
    
    ax1.plot(xvals_LCC_frame, yvals_LCC_frame, marker=',', c='k', lw=0, linestyle='')
    ax1.plot(xcirc, ycirc, c='r', lw=.5, label=r'$r_{\rm half-light, 0}$')
    
    Jacobi_radius = R_GC * (total_mass/galaxy_mass_list[j])**(1/3.)
    
    xJ = [ Jacobi_radius * np.cos(angle) for angle in angles ]
    yJ = [ Jacobi_radius * np.sin(angle) for angle in angles ]
    
    ax1.plot(xJ, yJ, c='g', lw=.5, label=r'$r_{tidal}$')
    
    ax1.legend(loc='center right', fontsize=4)
    
    ax1.set_xlim(-100., 100.)
    ax1.set_ylim(-100., 100.)
    
    ax1.annotate(r'$t_{\rm sim}\sim%.03f$ Myr'%(times[j]), xy=(0.5, 0.9), xycoords='axes fraction', fontsize=6)
    ax1.annotate(r'$M_{\rm LCC} = %.01f \, M_{\odot}$'%(total_mass), xy=(0.5, 0.85), xycoords='axes fraction', fontsize=6)
    ax1.set_xlabel(r'$(x-\tilde{x})_{\rm LCC}$ (pc)', fontsize=12)
    ax1.set_ylabel(r'$(y-\tilde{y})_{\rm LCC}$ (pc)', fontsize=12)
    
    ax2 = axs[1]#fig.add_subplot(122, adjustable='box-forced')#, sharex=ax1, sharey=ax1)
    ax2.set_yscale('log')
    ax2.set_xlim(0., 150.)
    ax2.set_ylim(1e-3, 5e0)
    ax2.plot(rvals_plot_naught, hist_naught, label=r'$n(r, t=0.0 \, {\rm Myr})$', linewidth=0.5)
    
    if j > 0:
        
        ax2.plot(rvals_plot, hist, label=r'$n(r, t\sim%.03f \, {\rm Myr})$'%(times[j]), linewidth=0.5)
    
    ax2.axvline(x=hl_radius, label=r'$r_{\rm half-light}(t=0.0 \, {\rm Myr})$', c='r', linewidth=0.5)
    ax2.axvline(x=Jacobi_radius, label=r'$r_{\rm tidal}(t\sim%.03f \, {\rm Myr})$'%(times[j]), c='g', linewidth=0.5)
    ax2.legend(loc='center right', fontsize=4)
    ax2.annotate(r'$n_{\rm today}(r) \propto \left(1 + \left(\frac{r}{a}\right)^{2}\right)^{-\gamma/2}$', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=6)
    ax2.annotate(r'$a = 50.1$ pc, $\gamma = 15.2$', xy=(0.3, 0.85), xycoords='axes fraction', fontsize=6)
    
    ax2.set_xlabel(r'$r_{\rm LCC}$ (pc)', fontsize=12)
    ax2.set_ylabel(r'$n(r) \hspace{2mm} [N_{\star}/{\rm pc}^{2}]$', fontsize=12)
    
    fig.suptitle('Lower Centaurus Crux model', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('LCC_only_%s.pdf'%(str(j).rjust(6, '0')))
    plt.close()