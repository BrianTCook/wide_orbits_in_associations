from __future__ import division, print_function
import operator
import time

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from amuse.lab import *
from amuse.couple import bridge
from amuse.units.optparse import OptionParser

from simulation_script import simulation

def main(Nstars, t_end, dt):

	#diskMass = diskMass|units.MSun
	#rMin = rMin|units.AU
	#rMax = rMax|units.AU
	t_end = t_end|units.Myr
	dt = dt|units.Myr

	simulation(Nstars, t_end, dt)
	#nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac, 

def new_option_parser():

	'''
	define an option parser for initial conditions
	'''

	optparser = OptionParser()
	#optparser.add_option('--nGas', dest='nGas', type='int', default=50000)
	optparser.add_option('--Nstars', dest='Nstars', type='int', default=1200)
	#optparser.add_option('--diskMass', dest='diskMass', type='float', default=1.0)
	#optparser.add_option('--rMin', dest='rMin', type='float', default=1.)
	#optparser.add_option('--rMax', dest='rMax', type='float', default=100.)
	#optparser.add_option('--Q', dest='Q', type='float', default=1.)
	#optparser.add_option('--diskmassfrac', dest='diskmassfrac', type='float', default=1.)
	
	optparser.add_option('--t_end', dest='t_end', type='float', default=20.)
	optparser.add_option('--dt', dest='dt', type='float', default=0.01) #megayears

	return optparser

if __name__ in '__main__':

	np.random.seed(42)


	o, arguments = new_option_parser().parse_args()
	main(**o.__dict__)
