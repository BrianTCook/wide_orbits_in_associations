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
    
def xyz_coords(a, gamma, Nstars):
    
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
    
    return xvals, yvals, zvals
    
    
def uvw_coords(sigma_u, sigma_v, sigma_w, Nstars):
    
    uvals = np.random.normal(loc=0., scale=sigma_U, size=(Nstars,))
    vvals = np.random.normal(loc=0., scale=sigma_U, size=(Nstars,))
    wvals = np.random.normal(loc=0., scale=sigma_U, size=(Nstars,))

    return uvals, vvals, wvals


def LCC_maker():
    
    stars = Particles(Nstars)
                    
    #LCC model using EFF formalism and measured velocity dispersions
    a, gamma = 50.1, 15.2
    sigma_U, sigma_V, sigma_W = 1.89, 0.9, 0.51
    
    Nstars = 1200
    
    xs, ys, zs = xyz_coords(a, gamma, Nstars)
    us, vs, ws = uvw_coords(sigma_u, sigma_v, sigma_w, Nstars)
    
    #Kroupa distribution, biggest stars are A-type stars
    masses = new_kroupa_mass_distribution(Nstars, 17.5|units.MSun)
    
    #no clumps yet
    #give each star appropriate phase space coordinates, mass
    
    for i, star in enumerate(stars):
        
        star.x = xs[i] | units.parsec
        star.y = xs[i] | units.parsec
        star.z = xs[i] | units.parsec
        
        star.vx = us[i] | units.kms
        star.vy = vs[i] | units.kms
        star.vz = ws[i] | units.kms
        
        star.mass = masses[i]

    return stars