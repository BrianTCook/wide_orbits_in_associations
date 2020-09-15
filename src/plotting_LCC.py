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

time_arrows = [ 'forward' ]

for arrow in time_arrows:
    
    phasespace_files = glob.glob(wide_orbits_direc + 'data/forward_in_time/phasespace_%s_*.ascii'%(arrow))
    stellar_files = glob.glob(wide_orbits_direc + 'data/forward_in_time/stellar_evolution_%s_*.ascii'%(arrow))
    times = np.loadtxt(wide_orbits_direc + 'data/forward_in_time/snapshot_times_%s.txt'%(arrow))
    galaxy_mass_list = np.loadtxt(wide_orbits_direc + 'data/forward_in_time/snapshot_galaxy_masses_%s.txt'%(arrow))
    
    hl_radius = 15.48
    angles = np.linspace(0., 2*np.pi, 100)
    xcirc = [ hl_radius * np.cos(angle) for angle in angles ]
    ycirc = [ hl_radius * np.sin(angle) for angle in angles ]
    
    for j, (file_phase, file_stellar) in enumerate(zip(phasespace_files, stellar_files)):
            
        print(j)
        time_show = times[j*20]
        
        data_phase = np.loadtxt(file_phase)
        data_stellar = np.loadtxt(file_stellar)
        
        total_mass = np.sum(data_phase[:,0])
    
        Nobjects = len(data_phase[:,0])
        xvals_stars = data_phase[:,1]
        yvals_stars = data_phase[:,2]
        zvals_stars = data_phase[:,3]
        
        x_med = np.median(xvals_stars)
        y_med = np.median(yvals_stars)
        z_med = np.median(zvals_stars)
        R_GC = np.sqrt(x_med**2. + y_med**2. + z_med**2.)
        
        lums = data_stellar[:,1]
        temps = data_stellar[:,2]
        
        plt.figure(figsize=(8,8))
        plt.gca().set_aspect('equal')
        
        plt.plot([x/1000. for x in xvals_stars], [y/1000. for y in yvals_stars], 
                 marker=',', c='k', lw=0, linestyle='')
        
        Jacobi_radius = R_GC/1000. * (total_mass/galaxy_mass_list[j])**(1/3.)
        
        xJ = [ x_med/1000. + Jacobi_radius * np.cos(angle) for angle in angles ]
        yJ = [ y_med/1000. + Jacobi_radius * np.sin(angle) for angle in angles ]
        
        plt.plot(xJ, yJ, c='r', lw=1, label=r'$r_{\rm tidal}$')
        
        plt.annotate(r'$t_{\rm sim}\simeq %.02f$ Myr'%(time_show), xy=(0.7, 0.9), xycoords='axes fraction', fontsize=16)
        
        plt.legend(loc='lower right', fontsize=16)
        
        plt.xlim(x_med/1000. - 0.1, x_med/1000. + 0.1)
        plt.ylim(y_med/1000. - 0.1, y_med/1000. + 0.1)
        plt.xlabel(r'$x_{\rm MW} \, ({\rm kpc})$', fontsize=16)
        plt.ylabel(r'$y_{\rm MW} \, ({\rm kpc})$', fontsize=16)
        plt.title(r'Lower Centaurus Crux, Milky Way Frame', fontsize=20)
        plt.tight_layout()
        plt.savefig('LCC_MW_%s.pdf'%(str(j).rjust(6, '0')))
        plt.close()
        
        '''
        masses = data_stellar[:,0]
        total_mass = np.sum(masses)
        
        x_med = np.median(xvals_stars)
        y_med = np.median(yvals_stars)
        z_med = np.median(zvals_stars)
        
        R_GC = np.sqrt(x_med**2. + y_med**2. + z_med**2.)
            
        xvals_LCC_frame = [ x-x_med for x in xvals_stars ]
        yvals_LCC_frame = [ y-y_med for y in yvals_stars ]
        zvals_LCC_frame = [ z-z_med for z in zvals_stars ]
            
        rvals_LCC_frame = [ np.sqrt(xvals_LCC_frame[i]**2. + yvals_LCC_frame[i]**2.) for i in range(Nobjects) ]
        
        hist, edges = np.histogram(rvals_LCC_frame, bins='auto')
        rvals_plot = [ 0.5*(edges[i]+edges[i+1]) for i in range(len(edges)-1) ]
        
        dr = rvals_plot[1] - rvals_plot[0]
        annuli_areas = [ 2 * np.pi * r * dr for r in rvals_plot ]
        
        hist = [ hist[i]/annuli_areas[i] for i in range(len(annuli_areas)) ]
            
        if arrow == 'forward':
            
            time_show = times[j*20]
            print(time_show)
            
        if arrow == 'backward':
        
            time_show = 16. - times[j]
            
        fig, axs = plt.subplots(ncols=2, figsize=(5,5))
        
        ax1 = axs[0]
        ax1.set_aspect('equal')
        
        ax1.plot(xvals_LCC_frame, yvals_LCC_frame, marker=',', c='k', lw=0, linestyle='')
        ax1.plot(xcirc, ycirc, c='r', lw=.5, label=r'$r_{\rm half-light, 0}$')
        
        Jacobi_radius = R_GC * (total_mass/galaxy_mass_list[j])**(1/3.)
        
        xJ = [ Jacobi_radius * np.cos(angle) for angle in angles ]
        yJ = [ Jacobi_radius * np.sin(angle) for angle in angles ]
        
        ax1.plot(xJ, yJ, c='g', lw=.5, label=r'$r_{\rm tidal}$')
        
        ax1.legend(loc='lower right', fontsize=5)
        
        ax1.set_xlim(-100., 100.)
        ax1.set_ylim(-100., 100.)
        
        ax1.set_title('Spatial Distribution', fontsize=10)
        ax1.annotate(r'$t_{\rm sim}\sim %.02f$ Myr'%(time_show), xy=(0.4, 0.92), xycoords='axes fraction', fontsize=5)
        ax1.annotate(r'$M_{\rm LCC} = %.01f \, M_{\odot}$'%(total_mass), xy=(0.4, 0.87), xycoords='axes fraction', fontsize=5)
        ax1.annotate(r'$N_{\star} = %i$'%(Nobjects), xy=(0.4, 0.82), xycoords='axes fraction', fontsize=5)
        ax1.set_xlabel(r'$(x-\tilde{x})_{\rm LCC}$ (pc)', fontsize=10)
        ax1.set_ylabel(r'$(y-\tilde{y})_{\rm LCC}$ (pc)', fontsize=10)
        plt.close()
        
        ax2 = axs[1]
        ax2.set_aspect('auto')
        
        ax2.set_yscale('log')
        ax2.set_xlim(0., 100.)
        ax2.set_ylim(1e-3, 5e0)
        ax2.plot(rvals_plot_naught, hist_naught, label=r'$n(r, t\sim 16.0 \, {\rm Myr})$', linewidth=0.5)
        
        if j != 160:
            
            ax2.plot(rvals_plot, hist, label=r'$n(r, t\sim %.02f \, {\rm Myr})$'%(time_show), linewidth=0.5)
        
        ax2.axvline(x=hl_radius, c='r', linewidth=0.5) #label=r'$r_{\rm half-light}(t\sim 16.0 \, {\rm Myr})$'
        ax2.axvline(x=Jacobi_radius, c='g', linewidth=0.5) #label=r'$r_{\rm tidal}(t\sim %.02f \, {\rm Myr})$'%(time_show)
        ax2.legend(loc='center right', fontsize=5)
        ax2.annotate(r'$n(r) \propto \left(1 + \left(\frac{r}{a}\right)^{2}\right)^{-\gamma/2}$', xy=(0.3, 0.85), xycoords='axes fraction', fontsize=5)
        ax2.annotate(r'$a = 50.1$ pc, $\gamma = 15.2$', xy=(0.3, 0.8), xycoords='axes fraction', fontsize=5)
        
        ax2.set_xlabel(r'$r_{\rm LCC}$ (pc)', fontsize=10)
        ax2.set_ylabel(r'$n(r) \hspace{2mm} [N_{\star}/{\rm pc}^{2}]$', fontsize=10)
        
        ax2.set_title('Number Density Profile', fontsize=10)
        fig.tight_layout()
        fig.suptitle('LCC model', fontsize=18)
        fig.subplots_adjust(top=0.88)
        fig.savefig('LCC_subplots_%s_%s.pdf'%(str(j).rjust(6, '0'), arrow))
        plt.close()
        '''