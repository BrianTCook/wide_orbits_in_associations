#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:51:30 2020

@author: BrianTCook
"""

from amuse.lab import *
from amuse.couple.bridge import CalculateFieldForParticles
from amuse.units import quantities
from amuse.community.bhtree.interface import BHTreeInterface, BHTree
from amuse.community.mercury.interface import Mercury
from amuse.community.huayno.interface import Huayno
from amuse.units import units,nbody_system

import numpy as np

from nemesis import *

'''
def getxv(converter, M1, a, e, ma=0):
    
    
    Get initial phase space coordinates (position and velocity) for an object around a central body
    
    converter - AMUSE unit converter
    M1        - Mass of the central body in AMUSE mass units
    a         - Semi-major axis of the orbit in AMUSE length units
    e         - Eccentricity of the orbit
    ma        - Mean anomaly of the orbit
    
    Returns: (x, v), the position and velocity vector of the orbit in AMUSE length and AMUSE length / time units
    
    kepler = Kepler(converter)
    kepler.initialize_code()
    kepler.initialize_from_elements(M1, a, e, mean_anomaly=ma) #Intoducing the attributes of the orbit
    
    x = quantities.as_vector_quantity(kepler.get_separation_vector()) #Grabbing the initial position and velocity
    v = quantities.as_vector_quantity(kepler.get_velocity_vector())
    
    kepler.stop()
    
    return x, v
 '''

def parent_worker():
    
    Mgalaxy, Rgalaxy = float(6.8e10)|units.MSun, 2.6|units.kpc #disk mass for MWPotential2014, Bovy(2015)
    converter_parent = nbody_system.nbody_to_si(Mgalaxy, Rgalaxy)
    code = BHTree(converter_parent) #done in src_nemesis
    
    #code.parameters.epsilon_squared=0.| units.kpc**2
    #code.parameters.end_time_accuracy_factor=0.
    #code.parameters.dt_param=0.1
    
    return code

def sub_worker(parts):
    
    #don't need parts as argument in the same way Simon did
    converter_sub = nbody_system.nbody_to_si(1000.|units.MSun, 10.|units.parsec) #masses list is in solar mass units
    code = Hermite(converter_sub) #Huayno might be too slow
    
    return code

def py_worker():
    
    code=CalculateFieldForParticles(gravity_constant = constants.G)
    
    return code

'''
also for nemesis
'''

def smaller_nbody_power_of_two(dt, conv):

    nbdt = conv.to_nbody(dt).value_in(nbody_system.time)
    idt = np.floor(np.log2(nbdt))

    return conv.to_si( 2**idt | nbody_system.time)

def radius(sys,eta=0.1,_G=constants.G): #eta=dt_param=0.1
    Mgalaxy, Rgalaxy = float(6.8e10)|units.MSun, 2.6|units.kpc #disk mass for MWPotential2014, Bovy(2015)
    converter_parent = nbody_system.nbody_to_si(Mgalaxy, Rgalaxy)
    dt=smaller_nbody_power_of_two(0.1 | units.Myr, converter_parent)
    radius=((_G*sys.total_mass()*dt**2/eta**2)**(1./3.))
    return radius*((len(sys)+1)/2.)**0.75

def timestep(ipart,jpart, eta=0.1/2,_G=constants.G): #eta=dt_param/2.=0.1/2.
    dx=ipart.x-jpart.x  
    dy=ipart.y-jpart.y
    dz=ipart.z-jpart.z
    dr2=dx**2+dy**2+dz**2
    dr=dr2**0.5
    dr3=dr*dr2
    mu=_G*(ipart.mass+jpart.mass)
    tau=eta/2./2.**0.5*(dr3/mu)**0.5
    return tau
