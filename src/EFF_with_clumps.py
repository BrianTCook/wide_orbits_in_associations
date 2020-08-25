#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:33:34 2020

@author: BrianTCook
"""

from __future__ import division, print_function
import time
import os
import math

#Circumvent a problem with using too many threads on OpenMPI
#os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = 'yes'

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from amuse.lab import *
from amuse.couple import bridge
from initial_conditions import initial_conditions

def EFF(r, a, gamma):
        
        return (1 + (r/a)**2.)**(-gamma/2.)
    
def enclosed_number(Nstars, r, a, gamma):
    
        #normalizing factor has to be 
    
        return Nstars * ( 1 - (1 - (r/a)**2)**(-gamma/2.+1) )
    
def xyz_coords(Nstars, Nclumps, a, gamma):

	rvals = []

	clump_populations = [ math.ceil(40.*np.random.random()) + 10 for i in range(Nclumps) ]

	print('clump populations: ', clump_populations)

	Nbins = 26
	radial_bins = np.linspace(0., 1.5*a, Nbins+1) #edges of bins in parsecs
	bin_populations =  [ 0 for i in range(Nbins) ]

	while len(rvals) < Nstars:

		whole_flag = 0
		clump_flag = 0

		while whole_flag == 0:

			rval_proposed = 1.5*a*np.random.rand()
			yval_a, yval_b = EFF(rval_proposed, a, gamma), np.random.rand()

			#rejection sampling
			if yval_b < yval_a:

				j = 0

				#finds the appropriate bin
				if radial_bins[j] > rval_proposed or radial_bins[j+1] <= rval_proposed:

					j += 1

				else:

					bin_middle = 0.5*(radial_bins[j]+radial_bins[j+1])

					#fills bin with a clump or just a single star
					if clump_flag < Nclumps:

						new_members = clump_populations[clump_flag]

					else:

						new_members = 1

					new_bin_population = bin_populations[j] + new_members

					if new_bin_population < enclosed_number(Nstars, bin_middle, a, gamma):

						rvals.append([rval_proposed for k in range(new_members)])
						phivals.append([2*np.pi*np.random.random() for k in range(new_members)])
						thetavals.append([np.arccos((2.*np.random.random()-1))for k in range(new_members)])

						bin_populations[j] += 1
						clump_flag += 1

						whole_flag = 1

				print('we should not be here')
				whole_flag = 1 #should not get here

	#unperturbed, clump members are right on top of each other
	rvals = [j for i in rvals for j in i]
	thetavals = [j for i in thetavals for j in i]
	phivals = [j for i in phivals for j in i]

    eps_x = 0.1 # pc
    eps_y = 0.1 # pc
    eps_z = 0.1 # pc

	xvals = [ rvals[i] * np.cos(phivals[i]) * np.sin(thetavals[i]) + eps_x * (np.random.rand() - 0.5) for i in range(Nstars) ]
	yvals = [ rvals[i] * np.sin(phivals[i]) * np.sin(thetavals[i]) + eps_y * (np.random.rand() - 0.5) for i in range(Nstars) ]
	zvals = [ rvals[i] * np.cos(thetavals[i]) + eps_z * (np.random.rand() - 0.5) for i in range(Nstars) ]

	return xvals, yvals, zvals
    
    
def uvw_coords(Nstars, sigma_u, sigma_v, sigma_w):
    
    uvals = np.random.normal(loc=0., scale=sigma_u, size=(Nstars,))
    vvals = np.random.normal(loc=0., scale=sigma_v, size=(Nstars,))
    wvals = np.random.normal(loc=0., scale=sigma_w, size=(Nstars,))

    return uvals, vvals, wvals

def LCC_maker(Nstars, Nclumps):
    
    stars = Particles(Nstars)
                    
    #LCC model using EFF formalism and measured velocity dispersions
    a, gamma = 50.1, 15.2
    sigma_u, sigma_v, sigma_w = 1.89, 0.9, 0.51
    
    xs, ys, zs = xyz_coords(Nstars, Nclumps, a, gamma)
    us, vs, ws = uvw_coords(Nstars, sigma_u, sigma_v, sigma_w)
    
    #Kroupa distribution, biggest stars are A-type stars
    masses = new_kroupa_mass_distribution(Nstars, 100.|units.MSun)
    
    #no clumps yet
    #give each star appropriate phase space coordinates, mass
    
    for i, star in enumerate(stars):
        
        star.x = xs[i] | units.parsec
        star.y = ys[i] | units.parsec
        star.z = zs[i] | units.parsec
        
        star.vx = us[i] | units.kms
        star.vy = vs[i] | units.kms
        star.vz = ws[i] | units.kms
        
        star.mass = masses[i]

    return stars
