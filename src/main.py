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

def main(mass_association, Nclumps):

	simulation(mass_association, Nclumps, time_reversal=True)

def new_option_parser():

	'''
	define an option parser for initial conditions
	'''

	optparser = OptionParser()
	optparser.add_option('--mass_association', dest='mass_association', type='float', default=700.)
	optparser.add_option('--Nclumps', dest='Nclumps', type='int', default=4)

	return optparser

if __name__ in '__main__':

	np.random.seed(42)
	o, arguments = new_option_parser().parse_args()
	main(**o.__dict__)
