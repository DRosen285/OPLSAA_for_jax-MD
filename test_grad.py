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
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.23)
#coulomb_handler = EwaldCoulomb(alpha=0.23, kmax=5, r_cut=cut_off_radius)

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

# --- Full breakdown for reporting/debugging ---
def energy_breakdown_full(R, nlist, coulomb_handler=None):
    E_bond, E_angle, E_torsion, E_improper, E_nb, E_bonded_lj = bonded_lj_fn_factory_full(R, nlist)

    e_real, e_recip, e_self, E_coulomb = coulomb_handler.energy(
        R, charges, displacement_fn, exclusion_mask, is_14_table, box_size, nlist
    )

    E_total = E_bonded_lj + E_coulomb

    return (
        E_bond,
        E_angle,
        E_torsion,
        E_improper,
        E_nb,
        e_real,
        e_recip,
        e_self,
        E_coulomb,
        E_total,
    )

# --- Lightweight total energy for MD/gradients ---
def total_energy_fn(R, nlist, coulomb_handler=None):
    _, _, _, _, _, E_bonded_lj = bonded_lj_fn_factory_full(R, nlist)

    _, _, _, E_coulomb = coulomb_handler.energy(
        R, charges, displacement_fn, exclusion_mask, is_14_table, box_size, nlist
    )

    return E_bonded_lj + E_coulomb

R_init = positions
nlist_test = neighbor_fn.update(R_init, neighbor_fn.allocate(R_init))


E_bond, E_angle, E_torsion, E_improper, E_nb,E_bonded_lj = bonded_lj_fn_factory_full(R_init, nlist_test)
e_real, e_reciprocal, e_self, E_coulomb = coulomb_handler.energy(R_init, charges, displacement_fn, exclusion_mask, is_14_table, box_size, nlist_test)


def bond_energy_fn(R):
    E_bond, *_ = breakdown_ener_jit(R, nlist_test, coulomb_handler=coulomb_handler)
    return E_bond


def angle_energy_fn(R):
    _, E_angle, *_ = breakdown_ener_jit(R, nlist_test, coulomb_handler=coulomb_handler)
    return E_angle

def torsion_energy_fn(R):
    _, _, E_torsion, *_  = breakdown_ener_jit(R, nlist_test, coulomb_handler=coulomb_handler)
    return E_torsion

def improper_energy_fn(R):
    _, _, _, E_improper, *_ = breakdown_ener_jit(R, nlist_test, coulomb_handler=coulomb_handler)
    return E_improper

def E_nb_only_fn(R):
    _, _, _, _, E_nb, *_ = breakdown_ener_jit(R, nlist_test, coulomb_handler=coulomb_handler)
    return E_nb

def E_coul(R):
    _, _, _, _, _, _, _, _, E_coulomb, _ = breakdown_ener_jit(R, nlist_test, coulomb_handler=coulomb_handler)
    return E_coulomb

# --- Force computation ---
def compute_forces(R, nlist, coulomb_handler):
    grad_E = grad_total_energy_fn(R, nlist, coulomb_handler)
    return -grad_E

breakdown_ener_jit=jit(energy_breakdown_full, static_argnames=["coulomb_handler"])
breakdown = breakdown_ener_jit(R_init,nlist_test,coulomb_handler=coulomb_handler)
(
    E_bond, E_angle, E_torsion, E_improper,
    E_nb, e_real, e_recip, e_self, E_coulomb, E_total
) = breakdown


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

grad_coul = grad(E_coul)(R_init)
print("NaNs in grad_coul:", jnp.isnan(grad_coul).any())

#e_tot_jit=jit(total_energy_fn, static_argnames=["coulomb_handler"])
# Get forces
grad_total_energy_fn = jit(
    jax.grad(total_energy_fn),
    static_argnames=["coulomb_handler"]
)
forces = compute_forces(R_init, nlist_test, coulomb_handler)
