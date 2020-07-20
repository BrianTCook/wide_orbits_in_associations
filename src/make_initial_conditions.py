from __future__ import division, print_function

import numpy as np

from amuse.lab import *
from amuse.ext.protodisk import ProtoPlanetaryDisk

def initial_conditions(nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac):
	
	'''
	nStars does nothing for now
	returns Sun, Neptune, pp disk w/ nGas particles and maximum radius rMax
	'''

	#for now, sun and Neptune
	stars_and_planets = new_solar_system()
	nParticles = len(stars_and_planets)
	Nep_ind = len(stars_and_planets) - 2 #avoids Pluto

	indices_to_keep = [ 0, Nep_ind ]

	stars_and_planets = [ stars_and_planets[i] for i in range(nParticles) if i in indices_to_keep ]

	#set up converter, pp disk gas particles
	np.random.seed(42)
	converter_gas = nbody_system.nbody_to_si(diskMass, rMax)
	gas = ProtoPlanetaryDisk(nGas, convert_nbody=converter_gas, Rmin=rMin, Rmax=rMax, q_out=Q, discfraction=diskmassfrac).result

	return stars_and_planets, gas
