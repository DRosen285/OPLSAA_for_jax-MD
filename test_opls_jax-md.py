import numpy as np
import jax.numpy as jnp
from jax_md import space
from jax_md import util
#from jax_md import partition
from jax_md.partition import neighbor_list
#from energy_oplsaa import opls_aa_neighbor_energy
from energy_oplsaa import opls_aa_energy, opls_aa_energy_with_nlist
from run_oplsaa_energy import parse_lammps_data

if __name__ == '__main__':
    # Load system
    positions, bonds, angles, torsions, impropers, nonbonded, box = parse_lammps_data('EC_data.txt')
    charges, sigmas, epsilons, pair_list, _ = nonbonded
    box_size = box

    energy_fn = opls_aa_energy(
        bonds, angles, torsions, impropers,
        nonbonded,
        box_size
    )

    energy_fn_nlist = opls_aa_energy_with_nlist(
        bonds, angles, torsions, impropers,
        nonbonded,
        box_size
    )

    # Compute energy
    
    energy = energy_fn(positions)
    print(f"Total OPLS-AA energy: {energy:.6f} kcal/mol")
    print(f"Total OPLS-AA energy: {energy:.6E} kcal/mol")

    energy_nlist = energy_fn_nlist(positions)
    print(f"Total OPLS-AA energy with nlist: {energy_nlist:.6f} kcal/mol")
    print(f"Total OPLS-AA energy with nlist: {energy_nlist:.6E} kcal/mol")

