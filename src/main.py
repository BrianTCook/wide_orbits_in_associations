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

def main(nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac, tEnd, dt):

	diskMass = diskMass|units.MSun
	rMin = rMin|units.AU
	rMax = rMax|units.AU
	tEnd = tEnd|units.yr
	dt = dt|units.yr

	simulation(nGas, nStars, diskMass, rMin, rMax, Q, diskmassfrac, tEnd, dt)

def new_option_parser():

	'''
	define an option parser for initial conditions
	'''

	optparser = OptionParser()
	optparser.add_option('--nGas', dest='nGas', type='int', default=10000)
	optparser.add_option('--nStars', dest='nStars', type='int', default=1)
	optparser.add_option('--diskMass', dest='diskMass', type='float', default=0.1)
	optparser.add_option('--rMin', dest='rMin', type='float', default=0.1)
	optparser.add_option('--rMax', dest='rMax', type='float', default=100.)
	optparser.add_option('--Q', dest='Q', type='float', default=1.)
	optparser.add_option('--diskmassfrac', dest='diskmassfrac', type='float', default=0.01)
	
	optparser.add_option('--tEnd', dest='tEnd', type='float', default=1000.) #years
	optparser.add_option('--dt', dest='dt', type='float', default=0.05) #years

	return optparser

if __name__ in '__main__':

	o, arguments = new_option_parser().parse_args()
	main(**o.__dict__)
