import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, grad,vmap
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
torsion_idx, k_torsion, _, n_torsion, gamma_torsion = torsions
improper_idx, k_improper, _, n_improper, gamma_improper = impropers



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
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.16219451)
#coulomb_handler = EwaldCoulomb(alpha=0.1540129, kmax=5, r_cut=cut_off_radius)

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


R_init = positions
nlist_test = neighbor_fn.update(R_init, neighbor_fn.allocate(R_init))


def bond_energy_fn(R):
    E_bond, _, _, _, _, _ = bonded_lj_fn_factory_full(R, neighbor_fn(R))
    return E_bond

def angle_energy_fn(R):
    _, E_angle, _, _, _, _ = bonded_lj_fn_factory_full(R, neighbor_fn(R))
    return E_angle

def torsion_energy_fn(R):
    _, _, E_torsion, _, _, _ = bonded_lj_fn_factory_full(R, neighbor_fn(R))
    return E_torsion

def improper_energy_fn(R):
    _, _, _, E_improper, _, _ = bonded_lj_fn_factory_full(R, neighbor_fn(R))
    return E_improper

def E_nb_only_fn(R):
    nlist = neighbor_fn(R)  # rebuild nlist from R
    _, _, _, _, E_nb, _ = bonded_lj_fn_factory_full(R, nlist)
    return E_nb

def E_bond_lj_total_fn(R):
    nlist = neighbor_fn(R)  # rebuild nlist from R
    _, _, _, _, _, E_total = bonded_lj_fn_factory_full(R, nlist)
    return E_total

def E_coul(R):
    nlist = neighbor_fn(R)
    _, _, _, E_coulomb = coulomb_handler.energy(R, charges, displacement_fn, exclusion_mask, is_14_table, box_size, nlist)
    return E_coulomb


grad_bond_test = grad(bond_energy_fn)(R_init)
print("NaNs in grad_bond_test:", jnp.isnan(grad_bond_test).any())

grad_angle_test = grad(angle_energy_fn)(R_init)
print("NaNs in grad_angle_test:", jnp.isnan(grad_angle_test).any())

grad_dihedral_test = grad(torsion_energy_fn)(R_init)
print("NaNs in grad_dihedral_test:", jnp.isnan(grad_dihedral_test).any())

grad_improper_test = grad(improper_energy_fn)(R_init)
print("NaNs in grad_improper_test:", jnp.isnan(grad_improper_test).any())

grad_nb = grad(E_nb_only_fn)(R_init)
print("NaNs in grad_nb:", jnp.isnan(grad_nb).any())

grad_total = grad(E_bond_lj_total_fn)(R_init)
print("NaNs in grad_total:", jnp.isnan(grad_total).any())

grad_coul = grad(E_coul)(R_init)
print("NaNs in grad_coul:", jnp.isnan(grad_coul).any())
