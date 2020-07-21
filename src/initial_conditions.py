from __future__ import division, print_function

import numpy as np

from amuse.lab import *
from amuse.ext.protodisk import ProtoPlanetaryDisk

def initial_conditions(nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac):
	
	'''
	nStars does nothing for now
	returns Sun, Jupiter, pp disk w/ nGas particles and maximum radius rMax
	'''

	#for now, sun and Neptune
	stars_and_planets = new_solar_system()
	Nep_ind = len(stars_and_planets) - 2 #avoids Pluto
	stars_and_planets = stars_and_planets.select(lambda n: n in ['SUN', 'JUPITER'], ['name'])

	#set up converter, pp disk gas particles
	np.random.seed(42)
	converter_gas = nbody_system.nbody_to_si(diskMass, 1.|units.AU)
	gas = ProtoPlanetaryDisk(nGas, convert_nbody=converter_gas, Rmin=rMin.value_in(units.AU), Rmax=rMax.value_in(units.AU), q_out=Q, discfraction=diskmassfrac).result

	return stars_and_planets, gas
