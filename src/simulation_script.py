from __future__ import division, print_function
import numpy as np

from amuse.lab import *
from make_initial_conditions import initial_conditions

def simulation(nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac, tEnd, dt):

	'''
	for now, just makes one star
	make a pp disk around it
	add a planet too

	run it using external and internal gravity solvers
	bridge them together
	'''

	stars_and_planets, gas = initial_conditions(nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac)

	external_bodies = stars_and_planets
	mass_external = external_bodies.mass.sum()
	a_init = 1.|units.AU
	converter_external = nbody_system.nbody_to_si(mass_external, a_init)
	gravity_external = ph4(converter_external)
	gravity_external.particles.add_particles(star_and_planet)
	
	internal_bodies = gas
	mass_internal = internal_bodies.mass.sum()
	Rmax = 20.|units.AU #pp disk max radius
	converter_internal = nbody_system.nbody_to_si(mass_internal, Rmax)
	gravity_internal = BHTree(converter_internal)
	gravity_internal.particles.add_particles(gas)

	channel_from_gravity_to_stars_and_planets = gravity.particles.new_channel_to(stars_and_planets)
	channel_from_gravity_gas = gravity.particles.new_channel_to(gas)

	sim_times_unitless = np.arange(0, tEnd, dt)
	sim_times = [ t|units.yr for t in sim_times_unitless ]

	for i, t in enumerate(sim_times):

		print(t)

		gravity.evolve_model(t)

	gravity.stop()

	return 1
