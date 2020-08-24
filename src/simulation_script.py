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
from EFF_with_clumps import LCC_maker

def simulation(Nstars, t_end, dt):

	'''
	for now, just makes one star
	make a pp disk around it
	add a planet too

	run it using external and internal gravity solvers
	bridge them together
	'''

	t0 = time.time()

	#stars_and_planets, gas = initial_conditions(nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac)
	stars_and_planets = LCC_maker(Nstars)

	eps = 1 | units.RSun

	mass_gravity = stars_and_planets.mass.sum()
	a_init = 10.|units.parsec
	converter_gravity = nbody_system.nbody_to_si(mass_gravity, a_init)
	
	gravity = ph4(converter_gravity)
	gravity.particles.add_particles(stars_and_planets)

	#internal_bodies = gas
	#mass_gas = gas.mass.sum()
	#converter_hydro = nbody_system.nbody_to_si(mass_gas, 1.|units.AU)

	#hydro = Fi(converter_hydro, mode='openmp')
	#hydro.gas_particles.add_particles(gas)

	gravity.parameters.epsilon_squared = eps**2
	#hydro.parameters.epsilon_squared = eps**2

	gravity_to_framework = gravity.particles.new_channel_to(stars_and_planets)
	#hydro_to_framework = hydro.gas_particles.new_channel_to(gas)

	#gravhydro = bridge.Bridge()
	#gravhydro.add_system(gravity, (hydro,))
	#gravhydro.add_system(hydro, (gravity,))
	#gravhydro.timestep = dt

	sim_times_unitless = np.arange(0., t_end.value_in(units.Myr), dt.value_in(units.Myr))
	sim_times = [ t|units.Myr for t in sim_times_unitless ]

	cm = plt.cm.get_cmap('rainbow')

	for j, t in enumerate(sim_times):

		if j%10 == 0:

			print('simulation time: ', t)
			print('wall time: %.02f minutes'%((time.time()-t0)/60.))
			print()

			#xvals_gas = gas.x.value_in(units.AU)
			#yvals_gas = gas.y.value_in(units.AU)

			xvals_stars_and_planets = stars_and_planets.x.value_in(units.parsec)
			yvals_stars_and_planets = stars_and_planets.y.value_in(units.parsec)

			#xy = np.vstack([xvals_gas, yvals_gas])
			#colors_gauss = gaussian_kde(xy)(xy)

			plt.figure()
			plt.gca().set_aspect('equal')
			#plt.scatter(xvals_gas, yvals_gas, s=6, marker='.', c=colors_gauss, cmap=cm, linewidths=0, label='Protoplanetary Disk')
			#plt.scatter(xvals_stars_and_planets[0], yvals_stars_and_planets[0], s=16, marker='*', c='k', label=r'Star ($M=M_{\odot}$)')
			#plt.scatter(xvals_stars_and_planets[0:], yvals_stars_and_planets[0:], s=16, marker='.', c='k', label=r'Gas Giants (Solar System)')
			plt.scatter(xvals_stars_and_planets, yvals_stars_and_planets, s=2, marker='*', c='k')

			plt.xlim(-30., 30.)
			plt.ylim(-30., 30.)
			plt.xlabel(r'$x$ (pc)', fontsize=12)
			plt.ylabel(r'$y$ (pc)', fontsize=12)
			plt.annotate(r'$t_{\rm sim} = %.02f$ Myr'%(t.value_in(units.Myr)), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8)
			plt.annotate(r'$M_{\rm LCC} = %.02f M_{\odot}$'%(stars_and_planets.mass.sum().value_in(units.MSun)), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
			plt.title('Lower Centaurus Crux model', fontsize=10)
			plt.tight_layout()
			plt.savefig('LCC_only_%s.png'%(str(j).rjust(5, '0')))
			plt.close()

		#gravhydro.evolve_model(t)
		gravity.evolve_model(t)
		gravity_to_framework.copy()
		#hydro_to_framework.copy()

	gravity.stop()
	hydro.stop()

	return 1

#extra stuff

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

'''
for i, t in enumerate(sim_times):

if i%10 == 0:

print('simulation time: %.02f yr'%(t.value_in(units.yr)))
print('wall time: %.02f minutes'%((time.time()-t0)/60.))
print('')

xvals = gravity.particles.x.value_in(units.AU)
yvals = gravity.particles.y.value_in(units.AU)

xvals_gas = xvals[:len(gas)]
yvals_gas = yvals[:len(gas)]

xvals_stars_and_planets = xvals[len(gas):]
yvals_stars_and_planets = yvals[len(gas):]

xy = np.vstack([xvals_gas, yvals_gas])
colors_gauss = gaussian_kde(xy)(xy)

plt.figure()
plt.gca().set_aspect('equal')
plt.scatter(xvals_gas, yvals_gas, s=10, marker='.', c=colors_gauss, cmap=cm, linewidths=0, label='Protoplanetary Disk')
plt.scatter(xvals_stars_and_planets, yvals_stars_and_planets, s=16, marker='*', c='k', label=r'Star ($M=M_{\odot}$)')
plt.xlim(-120., 120.)
plt.ylim(-120., 120.)
plt.xlabel(r'$x$ (AU)', fontsize=12)
plt.ylabel(r'$y$ (AU)', fontsize=12)
plt.annotate(r'$t_{\mathrm{sim}} = %.02f$ yr'%(t.value_in(units.yr)), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8)
plt.annotate(r'$M_{\mathrm{disk}} = %.02f M_{\odot}$'%(gas.mass.sum().value_in(units.MSun)), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
plt.legend(loc='lower right', fontsize=8)
plt.title('Young Protoplanetary Disk, Gravity + Hydrodynamics', fontsize=10)
plt.tight_layout()
plt.savefig('ppdisk_w_nothing_%s.png'%(str(i).rjust(5, '0')))
plt.close()

combined.evolve_model(t)
ch_gravity_to_gas.copy()
ch_gravity_to_stars_and_planets.copy()
ch_hydro_to_gas.copy()
'''
