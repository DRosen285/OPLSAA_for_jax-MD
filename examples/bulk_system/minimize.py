import jax
import jax.numpy as jnp
from jax import jit, lax, grad
from scipy.optimize import minimize as scipy_minimize
from jax_md import util, space, partition,minimize
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from extract_params_oplsaa import parse_lammps_data
from modular_Ewald import CutoffCoulomb, PME_Coulomb, EwaldCoulomb, make_is_14_lookup
from functools import partial
import time


# === Load system ===
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses = parse_lammps_data(
    'EC_bulk.data',
    'EC.settings'
)

charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded
bond_idx, k_b, r0 = bonds
angle_idx, k_theta, theta0 = angles

box_size = box

cut_off_radius = 15.0
dR = 0.5

displacement_fn, shift_fn = space.periodic(box)

# --- Neighbor list setup ---

neighbor_fn = partition.neighbor_list(
    displacement_fn, box, r_cutoff=cut_off_radius,
    dr_threshold=dR,
    mask=True,  # enables masking
    return_mask=True  # ensures mask is included in the returned NeighborList
    )



nlist = neighbor_fn.allocate(positions)

#for i in range(5):
#    print(f"Atom {i} neighbors:", nlist.idx[i])

##############################################

# === Coulomb Handler (Choose one) ===
#compute alpha based on defined precision
#coulomb_handler = CutoffCoulomb(r_cut=cut_off_radius)
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.16219451)
#coulomb_handler = EwaldCoulomb(alpha=0.1540129, kmax=5, r_cut=cut_off_radius)

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


# --- Lightweight total energy for MD/gradients ---
def make_energy_and_grad_fns(bonded_lj_factory, coulomb_handler,
                             charges, box_size, exclusion_mask,
                             is_14_table):
    """
    Returns:
      energy_fn(R, nlist) -> scalar energy
      grad_fn(R, nlist) -> dE/dR (same shape as R)
    Both are jitted and close over coulomb_handler (so it is static).
    """
    def energy_fn(R, nlist):
        # compute bonded+lj (factory returns energies when given nlist)
        _, _, _, _, _, E_bonded_lj = bonded_lj_factory(R, nlist)
        # coulomb returns (e_real, e_recip, e_self, total)
        _, _, _, E_coulomb = coulomb_handler.energy(
            R, charges, box_size, exclusion_mask, is_14_table, nlist
        )
        return E_bonded_lj + E_coulomb

    # JIT the energy and grad (coulomb_handler is closed over, so static)
    def neg_grad_fn(R, nlist):
        return -jax.grad(energy_fn)(R, nlist)
    energy_jit = jit(energy_fn)#,static_argnames=["coulomb_handler"])
    grad_jit = jit(jax.grad(energy_fn))#,static_argnames=["coulomb_handler"])
    neg_grad_jit = jit(neg_grad_fn)
    return energy_jit, grad_jit

energy_soft_jit, grad_soft_jit = make_energy_and_grad_fns(
    bonded_lj_fn_factory_soft, coulomb_handler,
    charges, box_size, exclusion_mask, is_14_table
)

energy_full_jit, grad_full_jit = make_energy_and_grad_fns(
        bonded_lj_fn_factory_full, coulomb_handler,
        charges, box_size, exclusion_mask, is_14_table
    )

# === Print initial energy ===

print("Initial energy ...")

energy_init= energy_full_jit(positions, nlist)#, coulomb_handler=coulomb_handler)

print(f"Total initial potential : {energy_init:.6f} kcal/mol")


# === jax-MD Minimizer ===


# --- Position wrapping ---
def wrap_positions(R, box_size):
    return R % box_size


# --- Optimizer setup ---
init_soft, apply_soft = minimize.fire_descent(energy_soft_jit, shift_fn, dt_start=1e-6,f_inc=1.03)
init_full, apply_full = minimize.fire_descent(energy_full_jit, shift_fn, dt_start=1e-5,f_inc=1.01, dt_max=0.01)
# --- initial opt state using soft minimizer (safe start) ---
opt_state = init_soft(positions, nlist=nlist)
mode = "soft"

# --- parameters: run soft first then full ---
n_soft_steps = 100        # run this many steps with soft LJ
n_full_steps = 2000       # then switch to full LJ
print_every = 50
neighbor_update_every = 1
force_tol = 1e-2         # tune as needed

def wrap_positions(R, box_size):
    return R % box_size

total_steps = n_soft_steps + n_full_steps
print("Minimizing (soft -> full).")
print("Step\tMode\tEnergy\tMax Force")
print("----------------------------------------")

for step in range(total_steps):
    R = wrap_positions(opt_state.position, box_size)

    if step % neighbor_update_every == 0:
        nlist = neighbor_fn.update(R, nlist)

    # switch to full mode at the right time
    if step == n_soft_steps:
        # re-init FIRE with the full energy using current positions
        opt_state = init_full(opt_state.position, nlist=nlist)
        mode = "full"
        print(f"Switched to FULL LJ at step {step}")

    # apply the correct optimizer
    if mode == "soft":
        opt_state = apply_soft(opt_state, nlist=nlist)
        energy = energy_soft_jit(R, nlist)
        force = grad_soft_jit(R, nlist)
    else:
        opt_state = apply_full(opt_state, nlist=nlist)
        energy = energy_full_jit(R, nlist)
        force = grad_full_jit(R, nlist)

    max_force = jnp.max(jnp.abs(force))

    if jnp.isnan(energy) or jnp.isnan(force).any():
        print(f"NaN detected at step {step}. Aborting minimization.")
        break

    if step % print_every == 0:
        print(f"{step}\t{mode}\t{float(energy):.6f}\t{float(max_force):.6f}")

    if max_force < force_tol:
        print(f"Converged at step {step} with max force {float(max_force):.6f}")
        break
