#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:26:02 2020

@author: BrianTCook
"""

import numpy as np
import math

np.random.seed(42)

Nsuns = 79

Nfirst = Nsuns
Nsecond = int(math.floor(0.25*Nsuns))
Nthird = int(math.floor(0.05*Nsuns))

print(Nfirst*4, Nsecond, Nthird)

first_limit, second_limit, third_limit, fourth_limit = 0.2, 4., 25., 60.

first_pop = np.random.uniform(low=first_limit, high=second_limit, size=(Nfirst,4))
first_pop_flattened = first_pop.flatten()
second_pop = np.random.uniform(low=second_limit, high=third_limit, size=(Nsecond,))
third_pop = np.random.uniform(low=third_limit, high=fourth_limit, size=(Nthird,))

first_mass = np.mean(first_pop) * Nfirst * 4
second_mass = np.mean(second_pop) * Nsecond
third_mass = np.mean(third_pop) * Nthird

Nplanets = (Nfirst*4) + Nsecond + Nthird

print('number of planets: %i'%(Nplanets))
print('average planetary mass: %.03f MJ'%((first_mass+second_mass+third_mass)/Nplanets))

ahh = np.concatenate([first_pop_flattened, second_pop, third_pop], axis=0)

plt.hist(ahh)

np.savetxt('first_population_planet_masses.txt', first_pop)
np.savetxt('second_population_planet_masses.txt', second_pop)
np.savetxt('third_population_planet_masses.txt', third_pop)