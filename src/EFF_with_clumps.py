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
from amuse.ic.salpeter import new_salpeter_mass_distribution
from initial_conditions import initial_conditions

def find_nearest_index(value, array):

	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()

	return idx

def EFF(r, a, gamma):
        
        return (1 + (r/a)**2.)**(-gamma/2.)

def HR_diagram_info(masses):

	lvals = []
	temps = []

	#a, b are coefficients for the mass-luminosity relation
	for mass in masses:

		if mass <= 0.45|units.MSun :

			a, b = 2.028, -0.976

		if mass > 0.45|units.MSun and mass <= 0.72|units.MSun :

			a, b = 4.572, -0.102

		if mass > 0.72|units.MSun and mass <= 1.05|units.MSun :

			a, b = 5.743, -0.007

		if mass > 1.05|units.MSun and mass <= 2.4|units.MSun :

			a, b = 4.329, 0.01

		if mass > 2.4|units.MSun and mass <= 7.|units.MSun :

			a, b = 3.967, 0.093

		if mass > 7.|units.MSun and mass <= 31.|units.MSun :

			a, b = 2.865, 1.105
		
		lum = 10**(a * np.log10(mass.value_in(units.MSun)) + b) #LSun

		if mass > 1.5|units.MSun:

			teff = 10**(-0.17*np.log10(mass.value_in(units.MSun))**2. + 0.888*np.log10(mass.value_in(units.MSun)) + 3.671)

		if mass <= 1.5|units.MSun:

			radius = (0.438*(mass.value_in(units.MSun))**2. + 0.479*mass.value_in(units.MSun) + 0.075)|units.RSun
			
			lum_units = lum|units.LSun

			sigma = 5.67e-8 | units.J * (units.s)**(-1.) * (units.m)**(-2.) * (units.K)**(-4.)

			teff = (lum_units / (4*np.pi*radius**2. * sigma))**(1/4.)
			teff = (teff.value_in(units.K))

		lvals.append(lum) #LSun
		temps.append(teff) #Kelvin

	return lvals, temps

def enclosed_number(Nstars, r, a, gamma):
    
        #normalizing factor has to be 
    
        return Nstars * ( 1 - (1 + (r/a)**2)**(-gamma/2.+1) )
    
def xyz_coords(Nstars, Nclumps, a, gamma):

	xvals, yvals, zvals = [], [], []

	clump_populations = [ math.ceil(40.*np.random.random() + 10.) for i in range(Nclumps) ]
	Nstars_in_clumps = np.sum(clump_populations)

	Nbins = 50
	bin_edges = np.linspace(0., 1.5*a, Nbins+1) #edges of bins in parsecs
	bin_centers = [ 0.5*(bin_edges[i] + bin_edges[i+1]) for i in range(Nbins) ]

	eps_x, eps_y, eps_z = 0.1, 0.1, 0.1 #parsecs, for perturbation purposes

	cdf = [ math.ceil(enclosed_number(Nstars, bin_centers[i], a, gamma)) for i in range(Nbins) ]

	allowances = [ 0 for i in range(Nbins) ]

	for i in range(Nbins):

		allowances[i] = int(cdf[i] - np.sum(allowances[:i]))	

	bin_populations =  [ 0 for i in range(Nbins) ]

	Nsystems = Nstars - Nstars_in_clumps + Nclumps

	clump_flag = 0

	while len(xvals) < Nsystems:

		rval_proposed = 1.5 * a * np.random.rand() #pc

		idx_bin = find_nearest_index(rval_proposed, bin_centers)	

		#fills bin with a clump or just a single star
		if clump_flag < Nclumps:

			new_members = clump_populations[clump_flag]

		else:

			new_members = 1

		new_bin_population = bin_populations[idx_bin] + new_members

		if new_bin_population < 1.05 * allowances[idx_bin]:

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

			bin_populations[idx_bin] = new_bin_population

	xvals = [ j for i in xvals for j in i ]
	yvals = [ j for i in yvals for j in i ]
	zvals = [ j for i in zvals for j in i ]

	return xvals, yvals, zvals
    
    
def uvw_coords(Nstars, sigma_u, sigma_v, sigma_w):
    
    uvals = np.random.normal(loc=0., scale=sigma_u, size=(Nstars,))
    vvals = np.random.normal(loc=0., scale=sigma_v, size=(Nstars,))
    wvals = np.random.normal(loc=0., scale=sigma_w, size=(Nstars,))

    return uvals, vvals, wvals

def LCC_maker(Nstars, Nclumps, time_reversal):
    
	stars = Particles(Nstars)
		    
	#LCC model using EFF formalism and measured velocity dispersions
	a, gamma = 50.1, 15.2
	sigma_u, sigma_v, sigma_w = 1.89, 0.9, 0.51

	xs, ys, zs = xyz_coords(Nstars, Nclumps, a, gamma)
	us, vs, ws = uvw_coords(Nstars, sigma_u, sigma_v, sigma_w)

	#Kroupa distribution, biggest stars are A-type stars
	masses = np.loadtxt('/home/brian/Desktop/wide_orbits_in_associations/data/LCC_masses.txt')

	#new_kroupa_mass_distribution(Nstars, 31.|units.MSun)

	masses = [ m|units.MSun for m in masses ]
	temps, lums = HR_diagram_info(masses)

	#np.savetxt('star_masses.txt', masses.value_in(units.MSun))
	#np.savetxt('star_temperatures.txt', temps)
	#np.savetxt('star_luminosities.txt', lums)

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
		star.luminosity = lums[i]
		star.temperature = temps[i]

	return stars
