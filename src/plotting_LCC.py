#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:22:20 2020

@author: BrianTCook
"""

			xvals_stars_and_planets = stars_and_planets.x.value_in(units.parsec)
			yvals_stars_and_planets = stars_and_planets.y.value_in(units.parsec)

			x_med = np.median(xvals_stars_and_planets)
			y_med = np.median(yvals_stars_and_planets)

			print('zeroth particle: x = %.02f pc, y = %.02f pc'%(xvals_stars_and_planets[0], yvals_stars_and_planets[0]))

			#xy = np.vstack([xvals_gas, yvals_gas])
			#colors_gauss = gaussian_kde(xy)(xy)

			plt.figure()
			plt.gca().set_aspect('equal')
			plt.plot([ x-x_med for x in xvals_stars_and_planets ], [ y-y_med for y in yvals_stars_and_planets ], marker='*', markersize=1, c='k', lw=0, linestyle='')

			plt.xlim(-100., 100.)
			plt.ylim(-100., 100.)

			plt.xlabel(r'$(x-\tilde{x})_{\rm LCC}$ (pc)', fontsize=12)
			plt.ylabel(r'$(y-\tilde{y})_{\rm LCC}$ (pc)', fontsize=12)
			plt.annotate(r'$t_{\rm sim} = %.02f$ Myr'%(t.value_in(units.Myr)), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8)
			plt.annotate(r'$M_{\rm LCC} = %.01f \, M_{\odot}$'%(stars_and_planets.mass.sum().value_in(units.MSun)), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
			plt.annotate(r'$\Sigma(r, t=0) \propto \left(1 + \left(\frac{r}{a}\right)^{2}\right)^{-\gamma/2}$', xy=(0.6, 0.95), xycoords='axes fraction', fontsize=8)
			plt.annotate(r'$a = 50.1$ pc', xy=(0.6, 0.9), xycoords='axes fraction', fontsize=8)
			plt.annotate(r'$\gamma = 15.2$', xy=(0.6, 0.85), xycoords='axes fraction', fontsize=8)
			plt.title('Lower Centaurus Crux (EFF) model', fontsize=10)
			plt.tight_layout()
			plt.savefig('LCC_only_%s.png'%(str(j).rjust(6, '0')))
			plt.close()