from galpy.potential import MWPotential2014
from galpy.util.bovy_conversion import get_physical, mass_in_msol
print(MWPotential2014)

#in units of kpc?
R_test = 8.
z_test = 0.001

m = 0.

for pot in MWPotential2014:

	m += pot.mass(R_test, z_test) * mass_in_msol(220., 8.)
	print(pot)
	print('enclosed mass: %.04e Msol'%(m))
