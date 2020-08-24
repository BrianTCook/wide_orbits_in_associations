#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:33:34 2020

@author: BrianTCook
"""

from __future__ import division, print_function
import time
import os

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
    
def xyz_coords(Nstars, a, gamma):
    
	rvals = []

	#rejection sampling
	while len(rvals) < Nstars:

		flag = 0

		while flag == 0:

			rval_proposed = 1.5*a*np.random.rand()
			yval_a, yval_b = EFF(rval_proposed, a, gamma), np.random.rand()

			if yval_b < yval_a:
		    
		    		rvals.append(rval_proposed)
		    		flag = 1
                    
	phivals = [ 2*np.pi*np.random.random() for i in range(Nstars) ]
	thetavals = [ np.arccos((2.*np.random.random()-1)) for i in range(Nstars) ]

	xvals = [ rvals[i] * np.cos(phivals[i]) * np.sin(thetavals[i]) for i in range(Nstars) ]
	yvals = [ rvals[i] * np.sin(phivals[i]) * np.sin(thetavals[i]) for i in range(Nstars) ]
	zvals = [ rvals[i] * np.cos(thetavals[i]) for i in range(Nstars) ]

	print(np.median(xvals), np.median(yvals))

	return xvals, yvals, zvals
    
    
def uvw_coords(Nstars, sigma_u, sigma_v, sigma_w):
    
    uvals = np.random.normal(loc=0., scale=sigma_u, size=(Nstars,))
    vvals = np.random.normal(loc=0., scale=sigma_v, size=(Nstars,))
    wvals = np.random.normal(loc=0., scale=sigma_w, size=(Nstars,))

    return uvals, vvals, wvals

def LCC_maker(Nstars):
    
    stars = Particles(Nstars)
                    
    #LCC model using EFF formalism and measured velocity dispersions
    a, gamma = 50.1, 15.2
    sigma_u, sigma_v, sigma_w = 1.89, 0.9, 0.51
    
    xs, ys, zs = xyz_coords(Nstars, a, gamma)
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
