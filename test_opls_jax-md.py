import numpy as np
import jax.numpy as jnp
from jax_md import space
from jax_md import util
#from jax_md import partition
from jax_md.partition import neighbor_list
#from energy_oplsaa import opls_aa_neighbor_energy
from energy_oplsaa import opls_aa_energy
from run_oplsaa_energy import parse_lammps_data

if __name__ == '__main__':
    # Load system
    positions, bonds, angles, torsions, impropers, nonbonded, box = parse_lammps_data('EC_data.txt')
    charges, sigmas, epsilons, pair_list, _ = nonbonded
    box_size = box

    # Create neighbor list
    displacement_fn, shift_fn = space.periodic(box_size)
    neighbor_fn = neighbor_list(displacement_fn, box_size, r_cutoff=1.2)
    nbrs = neighbor_fn(positions)

    # Create neighbor-compatible energy function
    #energy_fn = opls_aa_neighbor_energy(
    #    bonds, angles, torsions, impropers,
    #    charges, sigmas, epsilons,
    #    box_size, nbrs #neighbor_fn
    #)

    energy_fn = opls_aa_energy(
        bonds, angles, torsions, impropers,
        nonbonded,
        box_size
    )


    # Compute energy
    #energy = energy_fn(positions, nbrs)
 #   print(f"Total OPLS-AA energy using neighbor list: {energy:.6f} kJ/mol")
    energy = energy_fn(positions)
    print(f"Total OPLS-AA energy: {energy:.6f} kcal/mol")
    print(f"Total OPLS-AA energy: {energy:.6E} kcal/mol")

