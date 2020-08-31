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

phasespace_files = glob.glob('/Users/BrianTCook/Desktop/wide_orbits_in_associations/data/phasespace_*.ascii')


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
    
    half_light = plt.Circle((x_med, y_med), 15.48, color='r', fill=False)
    plt.gcf().gca().add_artist(half_light)
    
    ax1.set_xlim(-100., 100.)
    ax1.set_ylim(-100., 100.)
    
    #plt.annotate(r'$t_{\rm sim} = %.02f$ Myr'%(t.value_in(units.Myr)), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8)
    #plt.annotate(r'$M_{\rm LCC} = %.01f \, M_{\odot}$'%(stars_and_planets.mass.sum().value_in(units.MSun)), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
    ax1.set_xlabel(r'$(x-\tilde{x})_{\rm LCC}$ (pc)', fontsize=12)
    ax1.set_ylabel(r'$(y-\tilde{y})_{\rm LCC}$ (pc)', fontsize=12)
    
    ax2 = axs[1]#fig.add_subplot(122, adjustable='box-forced')#, sharex=ax1, sharey=ax1)
    ax2.set_yscale('log')
    ax2.set_xlim(0., 150.)
    ax2.set_ylim(1e-3, 1e1)
    ax2.plot(rvals_plot_naught, hist_naught, label=r'$n(r, t=0 {\rm Myr})$', linewidth=0.5)
    
    if j > 0:
        
        ax2.plot(rvals_plot, hist, label=r'$n(r, t= %i {\rm Myr})$'%(j), linewidth=0.5)
    
    ax2.axvline(x=15.48, label=r'$r_{\rm hl}(t=0)$', c='k', linewidth=0.5)
    ax2.legend(loc='center right', fontsize=6)
    ax2.annotate(r'$n_{\rm today}(r) \sim \left(1 + \left(\frac{r}{a}\right)^{2}\right)^{-\gamma/2}$', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=6)
    ax2.annotate(r'$a = 50.1$ pc, $\gamma = 15.2$', xy=(0.3, 0.85), xycoords='axes fraction', fontsize=6)
    
    ax2.set_xlabel(r'$(r-\tilde{r})_{\rm LCC}$ (pc)', fontsize=12)
    ax2.set_ylabel(r'$n(r) \hspace{2mm} [N_{\star}/{\rm pc}^{2}]$', fontsize=12)
    
    fig.suptitle('Lower Centaurus Crux model', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('LCC_only_%s.pdf'%(str(j).rjust(6, '0')))
    plt.close()