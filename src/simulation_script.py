from __future__ import division, print_function
import time
import os

#Circumvent a problem with using too many threads on OpenMPI
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = 'yes'

import numpy as np
from scipy.stats import gaussian_kde
import math
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from galpy.potential import MWPotential2014, to_amuse
from galpy.util.bovy_conversion import mass_in_msol

from amuse.lab import *
from amuse.couple import bridge
from amuse.support import io

from gravity_code import gravity_code_initial_setup, Hill_radius

def print_diagnostics(sim_time, t0, simulation_bodies, E_dyn, dE_dyn):
    
	print('------------')
	print('simulation time: ', sim_time)
	print('wall time: %.03f minutes'%((time.time() - t0)/60.))
	print('simulation_bodies.center_of_mass() in parsecs: ', simulation_bodies.center_of_mass().value_in(units.parsec))
	print('E_dyn: ', E_dyn)
	print('dE_dyn: %.04e'%(dE_dyn))
	print('------------')

def simulation(mass_association, Nclumps, time_reversal):

	'''
    runs gravity + stellar for the LCC model
	'''

    galaxy_code = to_amuse(MWPotential2014, t=0.0, tgalpy=0.0, reverse=False, ro=None, vo=None)
    
    code_name = 'not nemesis'
    stars, gravity, stellar = solver_codes_initial_setup(code_name, galaxy_code)
    
    channel_from_gravity_to_framework = gravity.particles.new_channel_to(stars)
    channel_from_framework_to_gravity = stars.new_channel_to(gravity.particles)
    
    channel_from_stellar_to_framework = stellar.particles.new_channel_to(stars)

	cluster_mass = stars.mass.sum()
	r_halflight = a * np.sqrt(4**(1./gamma) - 1.)
	r_virial = eta/6. * r_halflight

	t_dyn = np.sqrt(r_virial**3. / (constants.G * cluster_mass))

	print('-----------------------------------------')
	print('LCC dynamical time at present: %.04e Myr'%(t_dyn.value_in(units.Myr)))
	print('total_mass: %.03f MSun'%(np.sum(masses)))
	print('-----------------------------------------')

	t_dyn = (t_dyn.value_in(units.Myr))|units.Myr
    dt_parent = 2.|units.Myr

	t_backward = 16.|units.Myr
	t_forward = 48.|units.Myr

    sim_times_unitless = np.arange(0., (t_backward+t_forward).value_in(units.Myr), dt.value_in(units.Myr))
    sim_times = [ t|units.Myr for t in sim_times_unitless ]

	#for 3D numpy array storage
	Nsavetimes = len(sim_times)
	Ntotal = len(gravity.particles)
    
	all_data = np.zeros((Nsavetimes+1, Ntotal, 7))
	energy_data = np.zeros(Nsavetimes+1)
	time_data = np.zeros(Nsavetimes+1) 

	#for saving in write_set_to_file
	filename = 'data_temp.csv'
	attributes = ('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz')

	print('len(sim_times) is', len(sim_times))
	saving_flag = int(math.floor(len(sim_times)/Nsavetimes))

	snapshot_times = []
	snapshot_galaxy_masses = []
	j_like_index = 0

	if time_reversal == False:
		forward_or_backward = 'forward'
	else:
		forward_or_backward = 'backward'

	t0 = time.time()

	for j, t in enumerate(sim_times):

		energy = gravity.particles.potential_energy() + gravity.particles.kinetic_energy()
		deltaE = energy/energy_init - 1.

		print_diagnostics(t, t0, stars_and_planets, energy, deltaE)

		energy_data[j_like_index] = deltaE
		time_data[j_like_index] = t.value_in(units.Myr)

		j_like_index += 1
		io.write_set_to_file(gravity.particles, filename, 'csv',
				 attribute_types = (units.MSun, units.parsec, units.parsec, units.parsec, units.kms, units.kms, units.kms),
				 attribute_names = attributes)

		data_t = pd.read_csv(filename, names=list(attributes))
		data_t = data_t.drop([0, 1, 2]) #removes labels units, and unit names

		#data_t = data_t.drop(columns=['mass']) #goes from 7D --> 6D
		data_t = data_t.astype(float) #strings to floats

		all_data[j_like_index, :len(data_t.index), :] = data_t.values
		np.savetxt('phasespace_%s_frame_%s_LCC.ascii'%(forward_or_backward, str(j).rjust(5, '0')), data_t.values)

		snapshot_times.append(t.value_in(units.Myr))

		x_med, y_med, z_med = np.median(data_t['x']), np.median(data_t['y']), np.median(data_t['z'])

		Mgal = 0. #in solar masses
		Rgal, zgal = np.sqrt((x_med/1000.)**2. + (y_med/1000.)**2.), (z_med/1000.) #in kpc
		R_GC = np.sqrt(Rgal**2. + zgal**2.) #in kpc

		for pot in MWPotential2014:

			Mgal += pot.mass(Rgal, zgal) * mass_in_msol(220., 8.)

		snapshot_galaxy_masses.append(Mgal)

        stellar.evolve_model(t)
        channel_from_stellar_to_framework.copy()
        
        channel_from_framework_to_gravity.copy()
        gravity.evolve_model(t)
        channel_from_gravity_to_framework.copy()
        
	np.savetxt('snapshot_times_%s.txt'%(forward_or_backward), snapshot_times)
	np.savetxt('snapshot_galaxy_masses_%s.txt'%(forward_or_backward), snapshot_galaxy_masses)
    
	gravity.stop()
	stellar.stop()

	return 1