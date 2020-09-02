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

def main(Nstars, Nclumps):#, t_end, dt):

	#t_end = t_end|units.Myr
	#dt = dt|units.yr

	simulation(Nstars, Nclumps)#, t_end, dt, time_reversal=False)
	#simulation(Nstars, Nclumps, t_end, dt, time_reversal=True)

def new_option_parser():

	'''
	define an option parser for initial conditions
	'''

	optparser = OptionParser()
	optparser.add_option('--Nstars', dest='Nstars', type='int', default=1800)
	optparser.add_option('--Nclumps', dest='Nclumps', type='int', default=4)

	#optparser.add_option('--t_end', dest='t_end', type='float', default=16.)
	#optparser.add_option('--dt_tdyn_ratio', dest='dt_tdyn_ratio', type='float', default=5000.) #megayears

	return optparser

if __name__ in '__main__':

	np.random.seed(42)
	o, arguments = new_option_parser().parse_args()
	main(**o.__dict__)
