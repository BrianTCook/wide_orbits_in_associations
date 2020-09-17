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
from galpy.util import bovy_conversion

from amuse.lab import *
from amuse.couple import bridge
from amuse.support import io

from gravity_code import solver_codes_initial_setup, Hill_radius

def print_diagnostics(sim_time, t0, simulation_bodies, E_dyn, dE_dyn):
    
	print('------------')
	print('simulation time: ', sim_time)
	print('wall time: %.03f minutes'%((time.time() - t0)/60.))
	print('simulation_bodies.center_of_mass() in parsecs: ', simulation_bodies.center_of_mass().value_in(units.parsec))
	print('E_dyn: ', E_dyn)
	print('dE_dyn: %.04e'%(dE_dyn))
	print('------------')

def flip_velocity(x):

	try:

		x = str(float(x) * -1.)

	#rows with labels/units/etc.
	except:
		print('row with no numerical information')

	return x

def simulation(mass_association, Nclumps, time_reversal):

	'''
	runs gravity + stellar for the LCC model
	'''

	galaxy_code = to_amuse(MWPotential2014, t=0.0, tgalpy=0.0, reverse=False, ro=None, vo=None)
    
	code_name = 'not nemesis'
	stars_g, stars_s, gravity, stellar = solver_codes_initial_setup(code_name, galaxy_code, Nclumps, time_reversal) #stars for gravity, stars for stellar

	if time_reversal == False:

		forward_or_backward = 'forward'
		t_end, dt = 64.|units.Myr, 0.064|units.Myr

		sim_times_unitless = np.arange(0., (t_end+dt).value_in(units.Myr), dt.value_in(units.Myr))
		sim_times = [ t|units.Myr for t in sim_times_unitless ]

		#for 3D numpy array storage
		Nsavetimes = 8
		Ntotal = len(gravity.particles)
	    
		grav_data = np.zeros((Nsavetimes, Ntotal, 7))
		stellar_data = np.zeros((Nsavetimes, Ntotal, 3))
		energy_data = np.zeros(Nsavetimes+1)

		#for saving in write_set_to_file
		filename_grav = 'data_temp_grav.csv'
		filename_stellar = 'data_temp_stellar.csv'

		attributes_grav = ('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz')
		attributes_stellar = ('mass', 'luminosity', 'temperature')

		print('len(sim_times) is', len(sim_times))
		saving_flag = int(math.floor(len(sim_times)/Nsavetimes))
		print('saving_flag: %i'%(saving_flag))

		snapshot_galaxy_masses = [ 0. for i in range(Nsavetimes) ]
		snapshot_times = [ 0. for i in range(Nsavetimes) ]
		j_like_index = 0

		t0 = time.time()

		channel_from_gravity_to_framework = gravity.particles.new_channel_to(stars_g)
		channel_from_framework_to_gravity = stars_g.new_channel_to(gravity.particles)

		channel_from_stellar_to_framework = stellar.particles.new_channel_to(stars_s)

		energy_init = gravity.particles.potential_energy() + gravity.particles.kinetic_energy()

		for j, t in enumerate(sim_times):

			if j%saving_flag == 0:
	
				print('j = %i'%(j))

				energy = gravity.particles.potential_energy() + gravity.particles.kinetic_energy()
				deltaE = energy/energy_init - 1.

				print_diagnostics(t, t0, stars_g, energy, deltaE)

				energy_data[j_like_index] = deltaE

				#gravity stuff

				io.write_set_to_file(gravity.particles, filename_grav, 'csv',
						 attribute_types = (units.MSun, units.parsec, units.parsec, units.parsec, units.kms, units.kms, units.kms),
						 attribute_names = attributes_grav)

				data_t_grav = pd.read_csv(filename_grav, names=list(attributes_grav))
				data_t_grav = data_t_grav.drop([0, 1, 2]) #removes labels units, and unit names

				data_t_grav = data_t_grav.astype(float) #strings to floats

				grav_data[j_like_index, :len(data_t_grav.index), :] = data_t_grav.values
				np.savetxt('phasespace_%s_frame_%s_LCC.ascii'%(forward_or_backward, str(j).rjust(5, '0')), data_t_grav.values)

				#stellar stuff

				io.write_set_to_file(stellar.particles, filename_stellar, 'csv',
								 attribute_types = (units.MSun, units.LSun, units.K),
								 attribute_names = attributes_stellar)

				data_t_stellar = pd.read_csv(filename_stellar, names=list(attributes_stellar))
				data_t_stellar = data_t_stellar.drop([0, 1, 2]) #removes labels units, and unit names

				data_t_stellar = data_t_stellar.astype(float) #strings to floats

				stellar_data[j_like_index, :len(data_t_stellar.index), :] = data_t_stellar.values
				np.savetxt('stellar_evolution_%s_frame_%s_LCC.ascii'%(forward_or_backward, str(j).rjust(5, '0')), data_t_stellar.values)

				x_med, y_med, z_med = np.median(data_t_grav['x'])/1000., np.median(data_t_grav['y'])/1000., np.median(data_t_grav['z'])/1000.

				#compute MW mass at time t

				Mgal = 0. #in solar masses
				Rgal, zgal = np.sqrt(x_med**2. + y_med**2.), z_med #in kpc
				R_GC = np.sqrt(Rgal**2. + zgal**2.) #in kpc

				for pot in MWPotential2014:

					Mgal += pot.mass(Rgal, zgal) * bovy_conversion.mass_in_msol(220., 8.)

				snapshot_galaxy_masses[j_like_index] = Mgal #in MSun
				snapshot_times[j_like_index] = t
			
				j_like_index += 1

			stellar.evolve_model(t)
			channel_from_stellar_to_framework.copy()

			channel_from_framework_to_gravity.copy()
			gravity.evolve_model(t)
			channel_from_gravity_to_framework.copy()

		np.savetxt('snapshot_times_%s.txt'%(forward_or_backward), snapshot_times)
		np.savetxt('snapshot_galaxy_masses_%s.txt'%(forward_or_backward), snapshot_galaxy_masses)
		np.savetxt('snapshot_deltaEs_%s.txt'%(forward_or_backward), energy_data)

		gravity.stop()
		stellar.stop()

	else:

		forward_or_backward = 'backward'
		t_end, dt = 16.|units.Myr, 2|units.Myr

		channel_from_gravity_to_framework = gravity.particles.new_channel_to(stars_g)
		channel_from_framework_to_gravity = stars_g.new_channel_to(gravity.particles)

		gravity.timestep = dt

		print('backward simulation is ready to start')

		filename_init = 'LCC_phase_space_present_epoch.csv'
		attributes_grav = ('mass', 'x', 'y', 'z', 'vx', 'vy', 'vz')
		io.write_set_to_file(gravity.particles, filename_init, 'csv',
							 attribute_types = (units.MSun, units.parsec, units.parsec, units.parsec, units.kms, units.kms, units.kms),
							 attribute_names = attributes_grav)

		channel_from_framework_to_gravity.copy()
		gravity.evolve_model(t_end)
		channel_from_gravity_to_framework.copy()

		filename_backward = 'LCC_phase_space_backwards.csv'
		io.write_set_to_file(gravity.particles, filename_backward, 'csv',
					 attribute_types = (units.MSun, units.parsec, units.parsec, units.parsec, units.kms, units.kms, units.kms),
					 attribute_names = attributes_grav)

		gravity.stop()

		df_backwards = pd.read_csv(filename_backward, names=list(attributes_grav))

		df_ICs = df_backwards
		df_ICs['vx'].apply(flip_velocity)
		df_ICs['vy'].apply(flip_velocity)
		df_ICs['vz'].apply(flip_velocity)

		df_ICs.to_csv('LCC_phase_space_ICs.csv', index=False)
		
	return 1
