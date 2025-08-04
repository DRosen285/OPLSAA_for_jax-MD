# OPLSAA_for_jax-MD
Implementation of the OPLS-AA force field in jax-md

1.) run_oplsaa_energy.py: utilities to read force-field parameters/system setting from lammps input files (*.data, *.settings*)

2.) energy_oplsaa.py: computation of bonded and van der Waals interactions according to OPLS-AA

3.) modular_Ewald.py: computation of electrostatic interactions either via simple Coulomb equation, Ewald summation, or PME

4.) test_opls_jax-md.py: example how to compute the energy with jax-md using the custom OPLS-AA implementation

