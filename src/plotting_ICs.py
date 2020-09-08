#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 08:32:15 2020

@author: BrianTCook
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

datadir = '/Users/BrianTCook/Desktop/wide_orbits_in_associations/data'

df = pd.read_csv(datadir+'/temp_for_amuse.csv')
df = df.drop([0, 1])

masses = [ float(m) for m in df['#mass'].tolist() ]
total_mass = np.sum(masses)          


xvals = [ float(x) for x in df['x'].tolist() ]
yvals = [ float(y) for y in df['y'].tolist() ]

x_med = np.median(xvals)
y_med = np.median(yvals)

xvals = [ x - x_med for x in xvals ]
yvals = [ y - y_med for y in yvals ]

plt.figure(figsize=(6,6))
plt.gca().set_aspect('equal')
plt.plot(xvals, yvals, color='black',marker=',',lw=0, linestyle='')
plt.annotate(r'$M_{\rm LCC} = %.03f \hspace{2mm} M_{\odot}$'%(total_mass), xy=(0.6, 0.1), xycoords='axes fraction')
plt.annotate(r'$\rho(r) \sim (1 + (r/a)^{2})^{-\gamma/2}$', xy=(0.6, 0.05), xycoords='axes fraction')
plt.xlabel(r'$x-\tilde{x}_{\rm LCC}$ (pc)', fontsize=16)
plt.ylabel(r'$y-\tilde{y}_{\rm LCC}$ (pc)', fontsize=16)
plt.title(r'Lower Centaurus Crux at $t=0$ (sort of)', fontsize=16)
plt.tight_layout()
plt.savefig('LCC_IC.pdf')