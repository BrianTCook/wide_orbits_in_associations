#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:42:07 2020

@author: BrianTCook
"""

import matplotlib.pyplot as plt
import numpy as np

teff_vals = np.loadtxt('star_temperatures.txt')
lum_vals = np.loadtxt('star_luminosities.txt')

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

print(teff_vals)

plt.scatter(teff_vals, lum_vals)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.gca().invert_xaxis()