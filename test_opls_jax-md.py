import numpy as np
import jax.numpy as jnp
from jax_md import space
from jax_md import util
from energy_oplsaa import opls_aa_energy, opls_aa_energy_with_nlist_modular
from run_oplsaa_energy import parse_lammps_data
from modular_Ewald import CutoffCoulomb, EwaldCoulomb



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

    #coulomb_handler = CutoffCoulomb(r_cut=15.0, use_erfc=False, alpha=0.3)
    coulomb_handler = EwaldCoulomb(alpha=0.3, kmax=5, r_cut=15.0)


    energy_fn_nlist_modular = opls_aa_energy_with_nlist_modular(
        bonds, angles, torsions, impropers,
        nonbonded,
        box_size,
        coulomb_handler
    )


    # Compute energy
    
    energy = energy_fn(positions)
    print(f"Total OPLS-AA energy: {energy:.6f} kcal/mol")
    print(f"Total OPLS-AA energy: {energy:.6E} kcal/mol")

    energy_nlist_modular = energy_fn_nlist_modular(positions)
    print(f"Total OPLS-AA energy with nlist: {energy_nlist_modular:.6f} kcal/mol")
    print(f"Total OPLS-AA energy with nlist: {energy_nlist_modular:.6E} kcal/mol")


