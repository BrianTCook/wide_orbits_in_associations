import numpy
import threading

from amuse.datamodel import Particle,Particles,ParticlesOverlay
from amuse.units import units,nbody_system

from amuse.ext.basicgraph import UnionFind

def system_type(parts):
  if len(parts)==2:
    return "twobody"
  smass=sorted(parts.mass)
  if smass[-1]/smass[-2] > 10000.:
    return "solarsystem"
  return "nbody"    

class correction_from_compound_particles(object):
  def __init__(self, system, subsystems,worker_code_factory):
    self.system=system
    self.subsystems=subsystems
    self.worker_code_factory=worker_code_factory
    
  def get_gravity_at_point(self,radius,x,y,z):
    particles=self.system.copy()
    particles.ax=0. | (particles.vx.unit**2/particles.x.unit)
    particles.ay=0. | (particles.vx.unit**2/particles.x.unit)
    particles.az=0. | (particles.vx.unit**2/particles.x.unit)
    for parent in self.subsystems:
      sys=parent.subsystem 
      code=self.worker_code_factory()
      code.particles.add_particles(sys.copy())
      code.particles.position+=parent.position
      code.particles.velocity+=parent.velocity
      parts=particles-parent
      ax,ay,az=code.get_gravity_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.ax+=ax
      parts.ay+=ay
      parts.az+=az
      code=self.worker_code_factory()
      code.particles.add_particle(parent)
      ax,ay,az=code.get_gravity_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.ax-=ax
      parts.ay-=ay
      parts.az-=az
    return particles.ax,particles.ay,particles.az

  def get_potential_at_point(self,radius,x,y,z):
    particles=self.system.copy()
    particles.phi=0. | (particles.vx.unit**2)
    for parent in self.subsystems: 
      sys=parent.subsystem
      code=self.worker_code_factory()
      code.particles.add_particles(sys.copy())
      code.particles.position+=parent.position
      code.particles.velocity+=parent.velocity
      parts=particles-parent
      phi=code.get_potential_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.phi+=phi
      code=self.worker_code_factory()
      code.particles.add_particle(parent)
      phi=code.get_potential_at_point(0.*parts.radius,parts.x,parts.y,parts.z)
      parts.phi-=phi
    return particles.phi
  
class correction_for_compound_particles(object):  
  def __init__(self,system, parent, worker_code_factory):
    self.system=system
    self.parent=parent
    self.worker_code_factory=worker_code_factory
  
  def get_gravity_at_point(self,radius,x,y,z):
    parent=self.parent
    parts=self.system - parent
    instance=self.worker_code_factory()
    instance.particles.add_particles(parts)
    ax,ay,az=instance.get_gravity_at_point(0.*radius,parent.x+x,parent.y+y,parent.z+z)
    _ax,_ay,_az=instance.get_gravity_at_point([0.*parent.radius],[parent.x],[parent.y],[parent.z])
    instance.cleanup_code()
    return (ax-_ax[0]),(ay-_ay[0]),(az-_az[0])

  def get_potential_at_point(self,radius,x,y,z):
    parent=self.parent
    parts=self.system - parent
    instance=self.worker_code_factory()
    instance.particles.add_particles(parts)
    phi=instance.get_potential_at_point(0.*radius,parent.x+x,parent.y+y,parent.z+z)
    _phi=instance.get_potential_at_point([0.*parent.radius],[parent.x],[parent.y],[parent.z])
    instance.cleanup_code()
    return (phi-_phi[0])

class HierarchicalParticles(ParticlesOverlay):
  def __init__(self, *args,**kwargs):
    ParticlesOverlay.__init__(self,*args,**kwargs)
  def add_particles(self,sys):
    parts=ParticlesOverlay.add_particles(self,sys)
    if not hasattr(sys,"subsystem"):
      parts.subsystem=None
    return parts    
  def add_subsystem(self, sys, recenter=True):
    if len(sys)==1:
      return self.add_particles(sys)[0]
    p=Particle()
    self.assign_parent_attributes(sys, p, relative=False, recenter=recenter)
    parent=self.add_particle(p)
    parent.subsystem=sys
    return parent
  def assign_subsystem(self, sys, parent, relative=True, recenter=True):
    self.assign_parent_attributes(sys,parent,relative,recenter)
    parent.subsystem=sys
  def assign_parent_attributes(self,sys,parent, relative=True, recenter=True):
    parent.mass=sys.total_mass()
    if relative:
      pass
    else:
      parent.position=0.*sys[0].position
      parent.velocity=0.*sys[0].velocity
    if recenter:
      parent.position+=sys.center_of_mass()
      parent.velocity+=sys.center_of_mass_velocity()
      sys.move_to_center()
  def recenter_subsystems(self):
    for parent in self.compound_particles():
      parent.position+=parent.subsystem.center_of_mass()
      parent.velocity+=parent.subsystem.center_of_mass_velocity()
      parent.subsystem.move_to_center()  
  def compound_particles(self):
    return self.select( lambda x: x is not None, ["subsystem"])
  def simple_particles(self):
    return self.select( lambda x: x is None, ["subsystem"])
  def all(self):
    parts=Particles()
    for parent in self:
      if parent.subsystem is None:   
        parts.add_particle(parent)
      else:
        subsys=parts.add_particles(parent.subsystem)
        subsys.position+=parent.position
        subsys.velocity+=parent.velocity
    return parts


def kick_system(system, get_gravity, dt):
  parts=system.particles.copy()
  ax,ay,az=get_gravity(parts.radius,parts.x,parts.y,parts.z)
  parts.vx=parts.vx+dt*ax
  parts.vy=parts.vy+dt*ay
  parts.vz=parts.vz+dt*az
  channel = parts.new_channel_to(system.particles)
  channel.copy_attributes(["vx","vy","vz"])

def kick_particles(particles, get_gravity, dt):
  parts=particles.copy()
  ax,ay,az=get_gravity(parts.radius,parts.x,parts.y,parts.z)
  parts.vx=parts.vx+dt*ax
  parts.vy=parts.vy+dt*ay
  parts.vz=parts.vz+dt*az
  channel = parts.new_channel_to(particles)
  channel.copy_attributes(["vx","vy","vz"])

def potential_energy(system, get_potential):
  parts=system.particles.copy()
  pot=get_potential(parts.radius,parts.x,parts.y,parts.z)
  return (pot*parts.mass).sum()/2 

def potential_energy_particles(particles, get_potential):
  parts=particles.copy()
  pot=get_potential(parts.radius,parts.x,parts.y,parts.z)
  return (pot*parts.mass).sum()/2 

class Nemesis(object):
  def __init__(self,parent_code_factory,subcode_factory, worker_code_factory,
                use_threading=True):
    self.parent_code=parent_code_factory()
    self.subcode_factory=subcode_factory
    self.worker_code_factory=worker_code_factory
    self.particles=HierarchicalParticles(self.parent_code.particles)
    self.timestep=None
    self.subcodes=dict()
    self.split_treshold=None
    self.use_threading=use_threading
    self.radius=None

  def set_parent_particle_radius(self,p):
 
    if p.subsystem is None:
      sys=p.as_set()
    else:
      sys=p.subsystem
    
    if self.radius is None:
      p.radius=sys.virial_radius()
    else:
      if callable(self.radius): 
        p.radius=self.radius(sys)
      else:
        p.radius=self.radius

  def commit_particles(self):
    self.particles.recenter_subsystems()

    for p in self.particles:
      self.set_parent_particle_radius(p)
    
    if not hasattr(self.particles,"sub_worker_radius"):
      simple=self.particles.simple_particles()
      simple.sub_worker_radius=simple.radius #setting to 10 parsecs did not work well, this is probably a bad idea
      
    for parent in self.subcodes.keys():
      if parent.subsystem is self.subcodes[parent].particles:
        continue
      code=self.subcodes.pop(parent)
      del code
    for parent in self.particles.compound_particles():
      if parent not in subcodes:
        sys=parent.subsystem
        code=self.subcode_factory(sys)
        code.parameters.begin_time=self.model_time
        code.particles.add_particles(sys)
        parent.subsystem=code.particles
        self.subcodes[parent]=code

  def recommit_particles(self):
    self.commit_particles()

  def commit_parameters(self):
    pass

  def evolve_model(self, tend, timestep=None):
    if timestep is None:
      timestep = self.timestep
    if timestep is None:
      timestep = tend-self.model_time  
    while self.model_time < (tend-timestep/2.):    
      self.kick_codes(timestep/2.)
      self.drift_codes(self.model_time+timestep,self.model_time+timestep/2)
      self.kick_codes(timestep/2.)
      self.split_subcodes()

  def split_subcodes(self):
    subsystems=self.particles.compound_particles()
    to_remove=Particles()
    sys_to_add=[]
    for parent in subsystems:
      subsys=parent.subsystem
      radius=parent.radius
      components=subsys.connected_components(threshold=self.threshold,distfunc=self.distfunc)
      if len(components)>1:
        #print("splitting:", len(components))
        parentposition=parent.position
        parentvelocity=parent.velocity
        to_remove.add_particle(parent)
        for c in components:
          sys=c.copy()
          sys.position+=parentposition
          sys.velocity+=parentvelocity
          sys_to_add.append(sys)
        code=self.subcodes.pop(parent)
        del code  

    if len(to_remove)>0:
      self.particles.remove_particles(to_remove)

      for sys in sys_to_add:
        if len(sys)>1:
          newcode=self.subcode_factory(sys)
          newcode.parameters.begin_time=self.model_time
          newcode.particles.add_particles(sys)
          newparent=self.particles.add_subsystem(newcode.particles)
          newparent.sub_worker_radius=0.*newparent.radius
          self.subcodes[newparent]=newcode
        else:
          newparent=self.particles.add_subsystem(sys)
          newparent.sub_worker_radius=sys[0].radius
        self.set_parent_particle_radius(newparent)
        #print("radius in parsecs:",newparent.radius.in_(units.parsec))

      
  def handle_collision(self, coll_time,corr_time,coll_set):

    subsystems=self.particles.compound_particles()
    collsubset=self.particles[0:0]
    collsubsystems=Particles()
    for p in coll_set:
      p=p.as_particle_in_set(self.particles)
      collsubset+=p
      if p in self.subcodes:
        code=self.subcodes[p]
        code.evolve_model(coll_time)
      if p.subsystem is not None:
        collsubsystems.add_particle(p)

    #print("corr",coll_time.in_(units.yr),(coll_time-corr_time)/self.timestep)
    self.correction_kicks(collsubset,collsubsystems,coll_time-corr_time)
    
    newparts=HierarchicalParticles(Particles())
    to_remove=Particles()
    for p in coll_set:
      p=p.as_particle_in_set(self.particles)
      if p in self.subcodes:
        code=self.subcodes.pop(p)
        parts=code.particles.copy()
        parts.position+=p.position
        parts.velocity+=p.velocity
        newparts.add_particles(parts)
        del code
      else:
        np=newparts.add_particle(p)
        np.radius=p.sub_worker_radius        
      to_remove.add_particle(p)
    self.particles.remove_particles(to_remove)
    newcode=self.subcode_factory(newparts)
    newcode.parameters.begin_time=coll_time
    newcode.particles.add_particles(newparts)
    newparent=self.particles.add_subsystem(newcode.particles)
    self.set_parent_particle_radius(newparent)
    newparent.sub_worker_radius=0.*newparent.radius
    ##print("radius:",newparent.radius.in_(units.parsec),newparent.sub_worker_radius.in_(units.parsec))
    self.subcodes[newparent]=newcode
    return newparent
        
  def find_coll_sets(self,p1,p2):
    coll_sets=UnionFind()
    for p,q in zip(p1,p2):
      coll_sets.union(p,q)
    return coll_sets.sets()
    
  def drift_codes(self,tend,corr_time):
    code=self.parent_code
    stopping_condition = code.stopping_conditions.collision_detection
    stopping_condition.enable()
    while code.model_time < tend*(1-1.e-12):
      code.evolve_model(tend)
      if stopping_condition.is_set():
        coll_time=code.model_time
        #print("coll_time, t_end:", coll_time.in_(units.Myr), tend.in_(units.Myr))
        coll_sets=self.find_coll_sets(stopping_condition.particles(0), stopping_condition.particles(1))
        #print("collsets:",len(coll_sets))
        newparents=Particles()
        for cs in coll_sets:
          newparents.add_particle(self.handle_collision(coll_time,corr_time, cs))
        #print("len, corr-coll in Myr:",len(newparents),(corr_time-coll_time).in_(units.Myr))
        self.correction_kicks(self.particles,newparents,corr_time-coll_time)
        self.particles.recenter_subsystems()

          
    threads=[]
    for x in self.subcodes.values():
      threads.append(threading.Thread(target=x.evolve_model, args=(tend,)) )
    if self.use_threading:
      for x in threads: x.start()
      for x in threads: x.join()
    else:
      for x in threads: x.run()

  def kick_codes(self,dt):
    self.correction_kicks(self.particles,self.particles.select( lambda x: x is not None, ["subsystem"]),dt)
    self.particles.recenter_subsystems()

  def correction_kicks(self,particles,subsystems,dt):
    if len(subsystems)>0 and len(particles)>1:
      corrector=correction_from_compound_particles(particles,subsystems,self.worker_code_factory)
      kick_particles(particles,corrector.get_gravity_at_point, dt)

      corrector=correction_for_compound_particles(particles, None, self.worker_code_factory)
      for parent in subsystems:
        subsys=parent.subsystem
        corrector.parent=parent
        kick_particles(subsys, corrector.get_gravity_at_point, dt)

  def get_potential_at_point(self,radius,x,y,z):
    phi=self.parent_code.get_potential_at_point(radius,x,y,z)
    return phi

  def get_gravity_at_point(self,radius,x,y,z):
    ax,ay,az=self.parent_code.get_gravity_at_point(radius,x,y,z)
    return ax,ay,az

  @property
  def potential_energy(self):
    Ep=self.parent_code.potential_energy
    subsystems=self.particles.select( lambda x: x is not None, ["subsystem"])
    if len(self.particles)>1:
      corrector=correction_from_compound_particles(self.particles,
        subsystems,self.worker_code_factory)
      Ep+=potential_energy(self.parent_code,corrector.get_potential_at_point)    
    corrector=correction_for_compound_particles(self.particles, None, self.worker_code_factory)
    for parent,code in self.subcodes.items():
      Ep+=code.potential_energy
      if len(self.particles)>1:
        corrector.parent=parent
        Ep+=potential_energy(code,corrector.get_potential_at_point)
    return Ep

  @property
  def kinetic_energy(self):  
    Ek=self.parent_code.kinetic_energy
    for code in self.subcodes.values():
      Ek+=code.kinetic_energy
    return Ek

  @property
  def model_time(self):  
    return self.parent_code.model_time


