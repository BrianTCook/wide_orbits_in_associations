from __future__ import division, print_function
import operator
import time

import numpy as np
import matplotlib as plt
matplotlib.use('agg')
import matplotlib.pyplot as plt

from amuse.lab import *
from amuse.couple import bridge
from amuse.units.optparse import OptionParser

from simulation_script import simulation

def main(nGas, nStars, tEnd, dt):

	simulation(nGas, nStars, tEnd, dt)

def new_option_parser():

	'''
	define an option parser for initial conditions
	'''

	optparser = OptionParser()
	optparser.add_option('--nGas', dest='nGas', type='int', default=10000)
	optparser.add_option('--nStars', dest='nStars', type='int', default=1)
	optparser.add_option('--tEnd', dest='tEnd', type='float', default=100.) #years
	optparser.add_option('--dt', dest'dt', type='float', default=0.01) #years

	return optparser

if __name__ in '__main__':

	o, arguments = new_option_parser().parse_args()
	main(**o.__dict__)
