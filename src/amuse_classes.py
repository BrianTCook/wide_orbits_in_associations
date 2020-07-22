#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:45:17 2020

@author: BrianTCook
"""

import numpy as np
from amuse.lab import *

class BaseCode:
    
    def __init__(self, code, particles, eps=0|units.RSun):
        
        self.particles = particles
        m = self.particles.mass.sum()
        l = self.particles.position.length()
        self.converter = nbody_system.nbody_to_si(m, l)
        self.code = code(self.converter)
        self.code.parameters.epsilon_squared = eps**2
        
    def evolve_model(self, time):
        self.code.evolve_model(time)
    def copy_to_framework(self):
        self.channel_to_framework.copy()
    def get_gravity_at_point(self, r, x, y, z):
        return self.code.get_gravity_at_point(r, x, y, z)
    def get_potential_at_point(self, r, x, y, z):
        return self.code.get_potential_at_point(r, x, y, z)
    def get_timestep(self):
        return self.code.parameters.timestep
    
    @property
    def model_time(self):
        return self.code.model_time
    @property
    def particles(self):
        return self.code.particles
    @property
    def total_energy(self):
        return self.code.kinetic_energy + self.code.potential_energy
    @property
    def stop(self):
        return self.code.stop
    
class Gravity(BaseCode):
    
    def __init__(self, code, particles, eps=0|units.RSun):
        BaseCode.__init__(self, code, particles, eps)
        self.code.particles.add_particles(self.particles)
        self.channel_to_framework \
        = self.code.particles.new_channel_to(self.particles)
        self.channel_from_framework \
        = self.particles.new_channel_to(self.code.particles)
        self.initial_total_energy = self.total_energy
        
class Hydro(BaseCode):
    
    def __init__(self, code, particles, eps=0|units.RSun, dt=None, Rbound=None):
        BaseCode.__init__(self, code, particles, eps)
        self.channel_to_framework \
        = self.code.gas_particles.new_channel_to(self.particles)
        self.channel_from_framework \
        = self.particles.new_channel_to(self.code.gas_particles)
        self.code.gas_particles.add_particles(particles)
        m = self.particles.mass.sum()
        l = self.code.gas_particles.position.length()
        if Rbound is None:
            Rbound = 10*l
            self.code.parameters.periodic_box_size = Rbound
        if dt is None:
            dt = 0.01*np.sqrt(l**3/(constants.G*m))
            self.code.parameters.timestep = dt/8.
            self.initial_total_energy = self.total_energy
        
    @property
    def total_energy(self):
        return self.code.kinetic_energy \
        + self.code.potential_energy \
        + self.code.thermal_energy