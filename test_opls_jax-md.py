import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, grad
from scipy.optimize import minimize as scipy_minimize

from jax_md import space,partition
from jax_md import util

from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from run_oplsaa_multiple_molecules import parse_lammps_data
from modular_Ewald import CutoffCoulomb, PME_Coulomb, EwaldCoulomb, make_is_14_lookup


# === Load system ===
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box = parse_lammps_data(
    'EC.data',
    'EC.settings'
)

charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded
bond_idx, k_b, r0 = bonds
angle_idx, k_theta, theta0 = angles

box_size = box
cut_off_radius = 15.0
##### in case you want to print the neighbor list per atom for debugging purposes #####
dR=0.5
displacement_fn, shift_fn = space.periodic(box)
neighbor_fn = partition.neighbor_list(displacement_fn, box, r_cutoff=cut_off_radius, dr_threshold=dR, mask=True)
#positions = jnp.array(positions)
#nlist=neighbor_fn(positions)

#for i in range(5):
#    print(f"Atom {i} neighbors:", nlist.idx[i])

##############################################

# === Coulomb Handler (Choose one) ===
#compute alpha based on defined precision
#coulomb_handler = CutoffCoulomb(r_cut=cut_off_radius)
#coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.23)
coulomb_handler = EwaldCoulomb(alpha=0.23, kmax=5, r_cut=cut_off_radius)

# === Build neighbor list + displacement ===

bonded_lj_fn_factory_full, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id,box_size,
    use_soft_lj=False
)

# Precompute values
n_atoms = positions.shape[0]
is_14_table = make_is_14_lookup(pair_indices,is_14_mask, n_atoms)

# Exclusions
same_mol_mask = molecule_id[:, None] == molecule_id[None, :]  # (n_atoms, n_atoms)

exclusion_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)

# Apply exclusions for bonds and angles only if atoms are in the same molecule
bond_same_mol = molecule_id[bond_idx[:, 0]] == molecule_id[bond_idx[:, 1]]
angle_same_mol = molecule_id[angle_idx[:, 0]] == molecule_id[angle_idx[:, 2]]

bond_idx_filtered = bond_idx[bond_same_mol]
angle_idx_filtered = angle_idx[angle_same_mol]

exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 0], bond_idx_filtered[:, 1]].set(True)
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 1], bond_idx_filtered[:, 0]].set(True)

exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 0], angle_idx_filtered[:, 2]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 2], angle_idx_filtered[:, 0]].set(True)

# === Energy breakdown functions ===

def energy_breakdown_full(R, nlist, coulomb_handler=None):
    E_bond, E_angle, E_torsion, E_improper, E_nb,E_bonded_lj = bonded_lj_fn_factory_full(R, nlist)
    e_real, e_recip, e_self, E_coulomb = coulomb_handler.energy(R, charges, displacement_fn, exclusion_mask, is_14_table, box_size, nlist)
    return E_bond, E_angle, E_torsion, E_improper, E_nb, e_real, e_recip, e_self, E_coulomb, E_bonded_lj + E_coulomb


# JIT compile with static coulomb_handler
energy_breakdown_full_jit = jit(energy_breakdown_full, static_argnames=["coulomb_handler"])


# === Report energy ===
nlist_init = neighbor_fn.allocate(positions)
nlist_init = neighbor_fn.allocate(positions)
E_bond_init, E_angle_init, E_torsion_init, E_improper_init, E_nb_init, e_real_init, e_recip_init, e_self_init, E_coul_init, E_total_init = energy_breakdown_full_jit(positions, nlist_init,
                                                                                                             coulomb_handler=coulomb_handler)

print("\nEnergy terms:")
print(f"Bond            : {E_bond_init:.6f} kcal/mol")
print(f"Angle           : {E_angle_init:.6f} kcal/mol")
print(f"Torsion         : {E_torsion_init:.6f} kcal/mol")
print(f"Improper        : {E_improper_init:.6f} kcal/mol")
print(f"vdwl            : {E_nb_init:.6f} kcal/mol")
print(f"Coulomb_real    : {e_real_init:.6f} kcal/mol")
print(f"Coulomb_recip   : {e_recip_init:.6f} kcal/mol")
print(f"Coulomb_self    : {e_self_init:.6f} kcal/mol")
print(f"Coulomb total   : {E_coul_init:.6f} kcal/mol")
print(f"Total potential : {E_total_init:.6f} kcal/mol")


