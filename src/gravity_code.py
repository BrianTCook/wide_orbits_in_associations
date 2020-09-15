#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:28:08 2020

@author: BrianTCook
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:55:03 2020

@author: BrianTCook
"""

from amuse.lab import *
#from amuse.ext.bridge import bridge
from amuse.couple import bridge

import pandas as pd

from galpy.df import quasiisothermaldf
from galpy.potential import MWPotential2014, to_amuse
from galpy.util import bovy_conversion
from galpy.actionAngle import actionAngleStaeckel

from EFF_with_clumps import enclosed_mass, LCC_maker
from nemesis import *
from nemesis_supplement import *

import numpy as np
import time

def Hill_radius(r, Msub, Mparent):
    
    '''
    returns the radius at which the parent system exerts a nonnegligible influence
    '''
    
    return r * (Msub / (3*Mparent))**(1/3.)

def solver_codes_initial_setup(code_name, galaxy_code, Nclumps, time_reversal):
    
	'''
	will need to ask SPZ if he meant for field, orbiter to be separate in non
	Nemesis gravity solvers?
	'''

	if time_reversal == False:

		filename = '/home/brian/Desktop/wide_orbits_in_associations/data/LCC_phase_space_ICs.csv'
		stars = read_set_from_file(filename, "csv")

		#stored going the wrong way
		stars.vx *= -1.
		stars.vy *= -1.
		stars.vz *= -1.

	if time_reversal == True:
		mass_association = 700. #|units.MSun
		stars = LCC_maker(mass_association, Nclumps, time_reversal)

	x_med, y_med, z_med = np.median(stars.x.value_in(units.kpc)), np.median(stars.y.value_in(units.kpc)), np.median(stars.z.value_in(units.kpc))

	'''
	need to compute Hill radius for each star, and then use median for converter_sub units
	'''

	Mgal = 0. #in solar masses
	Rgal, zgal = np.sqrt(x_med**2. + y_med**2.), z_med #in kpc
	R_GC = np.sqrt(Rgal**2. + zgal**2.) #in kpc

	for pot in MWPotential2014:

		Mgal += pot.mass(Rgal, zgal) * bovy_conversion.mass_in_msol(220., 8.)

	converter_parent = nbody_system.nbody_to_si(Mgal|units.MSun, R_GC|units.kpc)
	converter_sub = nbody_system.nbody_to_si(np.median(stars.mass.value_in(units.MSun))|units.MSun, 1.|units.parsec) #masses list is in solar mass units

	gravity = bridge.Bridge(use_threading=False)

	if code_name != 'nemesis':

		stellar = SeBa()
		stellar.particles.add_particles(stars)

		#bridges each cluster with the bulge, not the other way around though
		herm = Hermite(converter_parent)
		herm.particles.add_particles(stars)
		gravity.add_system(herm, (galaxy_code,))  
	    
	if code_name == 'nemesis':

		all_bodies = stars

		'''
		need add_subsystem and assign_subsystem in HierarchicalParticles I think
		'''

		parts=HierarchicalParticles(all_bodies)

		dt=smaller_nbody_power_of_two(2.|units.Myr, converter_parent)
		print('dt_nemesis is %.04f Myr'%(dt.value_in(units.Myr)))

		nemesis=Nemesis(parent_worker, sub_worker, py_worker)
		nemesis.timestep=dt
		nemesis.distfunc=timestep
		nemesis.threshold=dt
		nemesis.radius=radius
		nemesis.commit_parameters()
		nemesis.particles.add_particles(parts)
		nemesis.commit_particles()

		stellar = SeBa()
		stellar.particles.add_particles(all_bodies)

		gravity.add_system(nemesis, (galaxy_code,))

	return gravity.particles, stellar.particles, gravity, stellar
