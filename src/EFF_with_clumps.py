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
from scipy.special import hyp2f1

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from amuse.lab import *
from amuse.couple import bridge
from amuse.ic.salpeter import new_salpeter_mass_distribution
from initial_conditions import initial_conditions

def find_nearest_index(value, array):

	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()

	return idx

def r_max_finder(mass_association, a, gamma):
    
    #normalizing factor has to be 
	min_star_mass = np.amin(new_kroupa_mass_distribution(1000, 17.5|units.MSun).value_in(units.MSun))

    delta_r = 1.|units.parsec

    matching_value = 1/3. * (a/delta_r) * min_star_mass / M_assoc
    
    def f(u):
        
        return 1/u * (1+u**(2.))**(gamma/2.)
    
    u_min, u_max = 1e-3, 20.
    
    while np.abs(u_max - u_min) < 1e-12:
        
        f1, f2 = f(u_min), f(u_max)
        
        if np.abs(f1 - matching_value) < np.abs(f2-matching_value):
            
            u_min = 0.5*(u_min + u_max)
            
        else:
            
            u_max = 0.5*(u_min + u_max)
            
    return 0.5*(u_min + u_max) * a.value_in(units.parsec)

def enclosed_mass(mass_association, r, a, gamma, r_max):
    
	rho_0 = 3 * mass_association / ( 4 * np.pi * r_max**3. ) #solar masses per parsec
	mass_enc = (4*np.pi / 3.) * rho_0 * r**(3.) * hyp2f1(3/2., (gamma+1.)/2., 5/2., -(r/a)**2.) #solar masses

	return mass_enc #no units, although we will need in terms of MSun
    
def xyz_coords(mass_association, Nclumps, a, gamma):

	xvals, yvals, zvals = [], [], []

	clump_populations = [ math.ceil(40.*np.random.random() + 10.) for i in range(Nclumps) ]
	Nstars_in_clumps = np.sum(clump_populations)

	Nbins = 500
    
	r_max = r_max_finder(mass_association, a, gamma)
	print('r_max: %.03f pc'%(r_max))
    
	bin_edges = np.linspace(0., r_max, Nbins+1) #edges of bins in parsecs
	bin_centers = [ 0.5*(bin_edges[i] + bin_edges[i+1]) for i in range(Nbins) ]

	eps_x, eps_y, eps_z = 0.1, 0.1, 0.1 #parsecs, for perturbation purposes

	cdf = [ enclosed_mass(mass_association, r, a, gamma, r_max) for r in bin_centers ]

	print('cdf all bins: ', cdf)

	allowances = [ 0 for i in range(Nbins) ]

	for i in range(Nbins):

		allowances[i] = int(cdf[i] - np.sum(allowances[:i]))	 #mass per slice allowed

	print('allowances in first 50 bins: ', allowances[:50])

	bin_masses =  [ 0. for i in range(Nbins) ]

	clump_flag = 0

	while np.sum(bin_masses) < mass_association:

		#print('assigned masses: %.03f MSun'%(np.sum(bin_masses)))
		rval_proposed = r_max * np.random.rand() #pc

		idx_bin = find_nearest_index(rval_proposed, bin_centers)	

		#fills bin with a clump or just a single star
		if clump_flag < Nclumps:

			new_members = clump_populations[clump_flag]

		else:

			new_members = 1

		new_member_masses = new_kroupa_mass_distribution(new_members, 17.5|units.MSun)
		new_bin_mass = bin_masses[idx_bin] + new_member_masses.sum().value_in(units.MSun)

		if new_bin_mass < 1.05 * allowances[idx_bin]:

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
			star_masses.append(new_member_masses)

			bin_masses[idx_bin] = new_bin_mass

	xvals = [ j for i in xvals for j in i ]
	yvals = [ j for i in yvals for j in i ]
	zvals = [ j for i in zvals for j in i ]
	star_masses = [ j for i in star_masses for j in i ]

	return xvals, yvals, zvals, star_masses
    
    
def uvw_coords(xs, ys, zs, sigma_squared_max, a):
    
	rs = [ np.sqrt(x**2. + y**2. + z**2.) for x, y, z in zip(xs, ys, zs) ]
	stdevs = [ np.sqrt(sigma_squared_max * r/a) for r in rs ] #standard deviation of velocity dispersions

	speeds = [ np.random.normal(loc=0., scale=std) for std in stdevs ] #in km/s

	Nstars = len(masses)

	random_directions = [ np.random.rand(3,) for i in range(Nstars) ]
	normalized_directions = [ direc/np.linalg.norm(direc) for direc in random_directions ]

	velocities = [ speeds[i] * normalized_directions[i] for i in range(Nstars) ]

	uvals = [ vel[0] for vel in velocities ]
	vvals = [ vel[1] for vel in velocities ]
	wvals = [ vel[2] for vel in velocities ]

	return uvals, vvals, wvals

def LCC_maker(mass_association, Nclumps, time_reversal):
    
		    
	#LCC model using EFF formalism and measured velocity dispersions
	a, gamma, sigma_squared_max = 50.1, 15.2, 2.15

    #Kroupa distribution, biggest stars are A-type stars
	xs, ys, zs, masses = xyz_coords(mass_association, Nclumps, a, gamma)
	us, vs, ws = uvw_coords(xs, ys, zs, sigma_squared_max, a)

	Nstars = len(masses)
	stars = Particles(Nstars)

	masses = [ m|units.MSun for m in masses ]
    
	#give each star appropriate phase space coordinates, mass

	#LCC coordinates in solar frame
	X0, Y0, Z0 = 57.7|units.parsec, -98.3|units.parsec, 16.6|units.parsec
	U0, V0, W0 = -8.96|units.kms, -20.55|units.kms, -6.29|units.kms

	#solar coordinates in MW frame
	XMW, YMW, ZMW = -8178.|units.parsec, 0.|units.parsec, 0|units.parsec
	UMW, VMW, WMW = 10.|units.kms, 247.4|units.kms, 0.|units.kms 

	if time_reversal == False:
		time_arrow = 1. 
	else:
		time_arrow = -1.

	for i, star in enumerate(stars):

		star.x = (xs[i] | units.parsec) + X0 + XMW
		star.y = (ys[i] | units.parsec) + Y0 + YMW
		star.z = (zs[i] | units.parsec) + Z0 + ZMW

		star.vx = time_arrow * ( (us[i] | units.kms) + U0 + UMW )
		star.vy = time_arrow * ( (vs[i] | units.kms) + V0 + VMW )
		star.vz = time_arrow * ( (ws[i] | units.kms) + W0 + WMW )

		star.mass = masses[i]

	return stars
