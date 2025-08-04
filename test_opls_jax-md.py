import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
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
coulomb_handler = CutoffCoulomb(r_cut=cut_off_radius)
#coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.23)
#coulomb_handler = EwaldCoulomb(alpha=0.23, kmax=5, r_cut=cut_off_radius)

# === Build neighbor list + displacement ===
bonded_lj_fn_factory_soft, neighbor_fn, displacement_fn = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id, box_size,
    use_soft_lj=True, lj_cap=1000.0
)

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

@jit
def energy_breakdown_soft(R, nlist):
    E_bonded_lj = bonded_lj_fn_factory_soft(R,nlist)
    e_real, e_recip, e_self, E_coulomb = coulomb_handler.energy(R, charges, displacement_fn, exclusion_mask, is_14_table, box_size, nlist)
    return E_bonded_lj, e_real, e_recip, e_self, E_coulomb, E_bonded_lj + E_coulomb

#@jit
def energy_breakdown_full(R, nlist,use_cut_off_only=False):
    E_bonded_lj = bonded_lj_fn_factory_full(R,nlist)
    #for cut-off
    if use_cut_off_only:
       E_coulomb = coulomb_handler.energy(positions, charges, displacement_fn, exclusion_mask, is_14_table, box)
       return E_bonded_lj, E_coulomb, E_bonded_lj + E_coulomb
    else:
       e_real, e_recip, e_self, E_coulomb = coulomb_handler.energy(R, charges, displacement_fn, exclusion_mask, is_14_table, box_size, nlist)
       return E_bonded_lj, e_real, e_recip, e_self, E_coulomb, E_bonded_lj + E_coulomb


# === Report energy ===
nlist_final = neighbor_fn.update(positions, neighbor_fn.allocate(positions))

E_bonded_min,E_coul_min, E_total_min = energy_breakdown_full(positions, nlist_final,use_cut_off_only=True)

print("\nAfter Minimization:")
print(f"Bonded+LJ       : {E_bonded_min:.6f} kcal/mol")
print(f"Coulomb total   : {E_coul_min:.6f} kcal/mol")
print(f"Total potential : {E_total_min:.6f} kcal/mol")


#E_bonded_min, e_real_min, e_recip_min, e_self_min, E_coul_min, E_total_min = energy_breakdown_full(positions, nlist_final)

#print("\nAfter Minimization:")
#print(f"Bonded+LJ       : {E_bonded_min:.6f} kcal/mol")
#print(f"Coulomb_real    : {e_real_min:.6f} kcal/mol")
#print(f"Coulomb_recip   : {e_recip_min:.6f} kcal/mol")
#print(f"Coulomb_self    : {e_self_min:.6f} kcal/mol")
#print(f"Coulomb total   : {E_coul_min:.6f} kcal/mol")
#print(f"Total potential : {E_total_min:.6f} kcal/mol")

