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
from amuse.community.nbody6xx.interface import Nbody6xx
from amuse.support import io

from initial_conditions import initial_conditions
from EFF_with_clumps import LCC_maker

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
	for now, just makes one star
	make a pp disk around it
	add a planet too

	run it using external and internal gravity solvers
	bridge them together
	'''

	a, gamma, eta = 50.1|units.parsec, 15.2, 9.1

	stars_and_planets = LCC_maker(mass_association, Nclumps, time_reversal)
	masses = stars_and_planets.mass.value_in(units.MSun)

	cluster_mass = stars_and_planets.mass.sum()
	r_halflight = a * np.sqrt(4**(1./gamma) - 1.)
	r_virial = eta/6. * r_halflight

	t_dyn = np.sqrt(r_virial**3. / (constants.G * cluster_mass))

	print('-----------------------------------------')
	print('LCC dynamical time at present: %.04e Myr'%(t_dyn.value_in(units.Myr)))
	print('total_mass: %.03f MSun'%(np.sum(masses)))
	print('-----------------------------------------')

	t_dyn = (t_dyn.value_in(units.Myr))|units.Myr

	eps = 1 | units.RSun

	mass_gravity = stars_and_planets.mass.sum()
	a_init = r_virial
	converter_gravity = nbody_system.nbody_to_si(mass_gravity, a_init)

	parent_code = Hermite(converter_gravity)
	dt_parent = 2.|units.Myr

	parent_code.particles.add_particles(stars_and_planets)
	parent_code.commit_particles()

	gravity = bridge.Bridge(use_threading=False)
	galaxy_code = to_amuse(MWPotential2014, t=0.0, tgalpy=0.0, reverse=False, ro=None, vo=None)
	gravity.add_system(parent_code, (galaxy_code,))

	gravity_to_framework = gravity.particles.new_channel_to(stars_and_planets)
	gravity.timestep = dt_parent

	t_backward = 16.|units.Myr
	t_forward = 48.|units.Myr

	gravity.evolve_model(t_backward)

	filename = 'temp_for_amuse'

	io.write_set_to_file(gravity.particles, filename, 'csv',
						 attribute_types = (units.MSun, units.parsec, units.parsec, units.parsec, units.kms, units.kms, units.kms),
						 attribute_names = attributes)

	data_t = pd.read_csv(filename, names=list(attributes))
	#data_t = data_t.drop([0, 1, 2]) #removes labels units, and unit names
	#data_t = data_t.astype(float) #strings to floats

	np.savetxt('phasespace_initial_LCC.ascii'%(forward_or_backward, str(j).rjust(5, '0')), data_t.values)

	'''
	#for 3D numpy array storage
	Nsavetimes = 50
	Ntotal = len(gravity.particles)
	all_data = np.zeros((Nsavetimes+1, Ntotal, 6))
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

		if j%saving_flag == 0:

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

			data_t = data_t.drop(columns=['mass']) #goes from 7D --> 6D
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

		#gravhydro.evolve_model(t)
		gravity.evolve_model(t)
		gravity_to_framework.copy()
		#hydro_to_framework.copy()

	#np.savetxt('snapshot_times_%s.txt'%(forward_or_backward), snapshot_times)
	#np.savetxt('snapshot_galaxy_masses_%s.txt'%(forward_or_backward), snapshot_galaxy_masses)

	np.savetxt('delta_energies_%i.txt'%(k), energy_data)
	np.savetxt('energy_times_%i.txt'%(k), time_data)

	gravity.stop()
	#hydro.stop()
	'''

	return 1

'''
external_bodies = stars_and_planets
mass_external = external_bodies.mass.sum()
a_init = 10.|units.AU
converter_external = nbody_system.nbody_to_si(mass_external, a_init)
gravity_external = ph4(converter_external)
gravity_external.particles.add_particles(stars_and_planets)

internal_bodies = gas
mass_internal = internal_bodies.mass.sum()
converter_internal = nbody_system.nbody_to_si(mass_internal, 1.|units.AU)
gravity_internal = Hermite(converter_internal)
gravity_internal.particles.add_particles(gas)

gravity = bridge.Bridge()
gravity.add_system(gravity_internal, (gravity_external,))
gravity.add_system(gravity_external, (gravity_internal,))	

ch_gravity_to_stars_and_planets = gravity.particles.new_channel_to(stars_and_planets)
ch_gravity_to_gas = gravity.particles.new_channel_to(gas)

hydro = Fi(converter_internal, mode='openmp')
hydro.gas_particles.add_particles(gas)

ch_hydro_to_gas = hydro.gas_particles.new_channel_to(gas)
ch_gas_to_hydro = gas.new_channel_to(hydro.gas_particles)

combined = bridge.Bridge(threading=False)
combined.add_system(gravity, (hydro,))
combined.add_system(hydro, (gravity,))

sim_times_unitless = np.arange(0, tEnd.value_in(units.yr), dt.value_in(units.yr))
sim_times = [ t|units.yr for t in sim_times_unitless ]
'''
