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
    
        return Nstars * ( 1 - (1 + (r/a)**2)**(-gamma/2.+1) )
    
def xyz_coords(Nstars, Nclumps, a, gamma):

	xvals, yvals, zvals = [], [], []

	clump_populations = [ math.ceil(5.*np.random.random()) for i in range(Nclumps) ]
	Nstars_in_clumps = np.sum(clump_populations)

	Nbins = 10
	radial_bins = np.linspace(0., 1.5*a, Nbins+1) #edges of bins in parsecs

	eps_x, eps_y, eps_z = 0.1, 0.1, 0.1 #parsecs, for perturbation purposes

	cdf = [ math.ceil(enclosed_number(Nstars, 0.5*(radial_bins[i] + radial_bins[i+1]), a, gamma)) for i in range(Nbins) ]

	allowances = [ 0 for i in range(Nbins) ]

	for i in range(Nbins):

		allowances[i] = int(cdf[i] - np.sum(allowances[:i]))	

	bin_populations =  [ 0 for i in range(Nbins) ]

	Nsystems = Nstars - Nstars_in_clumps + Nclumps

	clump_flag = 0

	while len(xvals) < Nsystems:

		rval_proposed = 1.5 * a * np.random.rand() #pc

		j = 0

		#finds the appropriate bin
		if radial_bins[j] < rval_proposed and radial_bins[j+1] >= rval_proposed:

			bin_middle = 0.5*(radial_bins[j]+radial_bins[j+1])

			#fills bin with a clump or just a single star
			if clump_flag < Nclumps:

				new_members = clump_populations[clump_flag]

			else:

				new_members = 1

			new_bin_population = bin_populations[j] + new_members

			if new_bin_population < allowances[j]:

				if clump_flag < Nclumps:
					clump_flag += 1

				rvals_new = [rval_proposed for k in range(new_members)]
				phivals_new = [2*np.pi*np.random.random() for k in range(new_members)]
				thetavals_new = [np.arccos((2.*np.random.random()-1))for k in range(new_members)]

				xvals_new = [ rvals_new[k] * np.cos(phivals_new[k]) * np.sin(thetavals_new[k]) for k in range(new_members) ]
				yvals_new = [ rvals_new[k] * np.sin(phivals_new[k]) * np.sin(thetavals_new[k]) for k in range(new_members) ]
				zvals_new = [ rvals_new[k] * np.cos(thetavals_new[k]) for k in range(new_members) ]

				#randomly perturb cluster members
				if new_members > 1:
				    
					xvals_new = [ x + eps_x * np.random.rand() for x in xvals_new ]
					yvals_new = [ y + eps_y * np.random.rand() for y in yvals_new ]
					zvals_new = [ z + eps_z * np.random.rand() for z in zvals_new ]

				xvals.append(xvals_new)
				yvals.append(yvals_new)
				zvals.append(zvals_new)

				print(xvals)

				bin_populations[j] += new_members
		else:

			j += 1

		if j >= Nbins-1:

			print('we should not be here')
			return
    
	xvals = [ j for i in xvals for j in i ]
	yvals = [ j for i in yvals for j in i ]
	zvals = [ j for i in zvals for j in i ]

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
