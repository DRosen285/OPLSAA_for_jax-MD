#Nose-Hoover thermostat for NVT simulations
import jax
import jax.numpy as jnp
from jax import jit, lax, grad
from jax import random
from scipy.optimize import minimize as scipy_minimize
from jax_md import util, space, partition, minimize
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from extract_params_oplsaa import parse_lammps_data
from modular_Ewald import CutoffCoulomb, PME_Coulomb, EwaldCoulomb, make_is_14_lookup

from jax_md import units, quantity
from jax_md import simulate
from functools import partial

# === Load system ===
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses = parse_lammps_data(
    'system_after_lammps_min.data',
    'system.settings'
)

unit = units.real_unit_system()

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
    mask=True,
    return_mask=True
)

nlist = neighbor_fn.allocate(positions)

# === Coulomb Handler ===
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.16219451)

# === Build force field ===
bonded_lj_fn_factory_full, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id, box_size,
    use_soft_lj=False
)

n_atoms = positions.shape[0]
is_14_table = make_is_14_lookup(pair_indices, is_14_mask, n_atoms)

# Exclusions
same_mol_mask = molecule_id[:, None] == molecule_id[None, :]
exclusion_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)

bond_same_mol = molecule_id[bond_idx[:, 0]] == molecule_id[bond_idx[:, 1]]
angle_same_mol = molecule_id[angle_idx[:, 0]] == molecule_id[angle_idx[:, 2]]

bond_idx_filtered = bond_idx[bond_same_mol]
angle_idx_filtered = angle_idx[angle_same_mol]

exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 0], bond_idx_filtered[:, 1]].set(True)
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 1], bond_idx_filtered[:, 0]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 0], angle_idx_filtered[:, 2]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 2], angle_idx_filtered[:, 0]].set(True)


# --- Energy + Grad factories ---

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


energy_full_jit, grad_full_jit = make_energy_and_grad_fns(
        bonded_lj_fn_factory_full, coulomb_handler,
        charges, box_size, exclusion_mask, is_14_table
    )


print("Initial energy ...")
energy_init= energy_full_jit(positions, nlist)
print(f"Total initial potential : {energy_init:.6f} kcal/mol")

# === NVT Nose-Hoover ===
timestep = 1
fs = timestep * unit['time']
print(fs,unit['time'])
dt = fs
tau_T=100.0 * unit['time']
write_every = 100
T_init = 298 * unit['temperature']
print(unit['temperature'])
key = random.PRNGKey(121) #used to initialize velocities from Maxwell-Boltzmann distribution
steps = 1000
mass = jnp.array(masses).reshape(-1) *unit['mass'] 
init, apply =  simulate.nvt_nose_hoover(
    energy_full_jit,
    shift_fn,
    dt=dt,
    kT=T_init,
    tau=tau_T,
    mass=mass
)

state = init(key, positions, nlist=nlist)

# === Logging setup ===
log = {
    'kT': jnp.zeros((steps,)),
    'KE': jnp.zeros((steps,)),
    'PE': jnp.zeros((steps,)),
    'position': jnp.zeros((steps // write_every,) + positions.shape)
}

# === Step function with correct broadcasting ===
def step_fn(carry, i):
    state, nlist, log = carry

    # Momentum with proper mass broadcasting
    mom = state.velocity * mass[:, None]

    T = quantity.temperature(momentum=mom, mass=jnp.array(masses).reshape(-1, 1))
    KE = quantity.kinetic_energy(momentum=mom, mass=jnp.array(masses).reshape(-1, 1))
    PE = energy_full_jit(state.position,nlist)

    log['kT'] = log['kT'].at[i].set(T)
    log['KE'] = log['KE'].at[i].set(KE)
    log['PE'] = log['PE'].at[i].set(PE)

    log['position'] = lax.cond(
        i % write_every == 0,
        lambda p: p.at[i // write_every].set(state.position),
        lambda p: p,
        log['position']
    )

    state = apply(state,nlist=nlist)
    nlist = nlist.update(state.position)

    return (state, nlist, log), None

(state, nlist, log), _ = lax.scan(step_fn, (state, nlist, log), jnp.arange(steps))

R = state.position
#print("Final energy ...")
energy_final = energy_full_jit(R, nlist)
print(f"Total final potential : {energy_final:.6f} kcal/mol")

