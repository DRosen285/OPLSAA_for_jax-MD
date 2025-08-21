import jax
# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, lax, random
from jax_md import units, quantity, simulate, space, partition
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from run_oplsaa_multiple_molecules import parse_lammps_data
from modular_Ewald import PME_Coulomb, make_is_14_lookup
from jax_md.util import f64

# === Load system ===
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses = parse_lammps_data(
    'system_after_lammps_min.data',
    'system.settings'
)
unit = units.real_unit_system()
charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded
bond_idx, k_b, r0 = bonds
angle_idx, k_theta, theta0 = angles
# --- Neighbor list ---
cut_off_radius = 15.0
dR = 0.5
box_matrix = jnp.diag(box)
disp_fn_init, shift_fn_init = space.periodic_general(box_matrix,fractional_coordinates=False)

neighbor_fn = partition.neighbor_list(
    displacement_or_metric=disp_fn_init,
    box=box_matrix,
    r_cutoff=cut_off_radius,
    dr_threshold=dR,
    mask=True,
    return_mask=True
)
# allocate with the initial box explicitly
nlist = neighbor_fn.allocate(positions,box=box_matrix)

# --- Coulomb handler ---
coulomb_handler = PME_Coulomb(grid_size=16, alpha=0.16219451)  # reduced grid size for CPU

# --- Force field ---
bonded_lj_fn_factory_full, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id, box,
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

# --- Energy function with dynamic box ---
def make_energy_fn_dynamic_box(bonded_lj_factory, coulomb_handler, charges, exclusion_mask, is_14_table):
    def energy_fn(R, nlist, box, **kwargs):
        box_matrix = jnp.diag(box)
        disp_fn, _ = space.periodic_general(box_matrix)
        *_, E_bonded_lj = bonded_lj_factory(R, nlist)
        _, _, _, E_coulomb = coulomb_handler.energy(
            R, charges, box, exclusion_mask, is_14_table, nlist
        )
        return E_bonded_lj + E_coulomb
    return jax.jit(energy_fn)

energy_full_jit = make_energy_fn_dynamic_box(
    bonded_lj_fn_factory_full, coulomb_handler, charges, exclusion_mask, is_14_table
)

print("Initial energy ...")
energy_init = energy_full_jit(positions, nlist, box)
print(f"Total initial potential : {energy_init:.6f} kcal/mol")

# --- Stabilized NPT parameters ---
dt = 0.5 * unit['time']
tau_T = 500.0 * unit['time']
tau_P = 5000.0 * unit['time']
T_init = 298.0 * unit['temperature']
P_init = 1.0 * unit['pressure']
key = random.PRNGKey(121)
steps = 1000
mass = jnp.array(masses).reshape(-1) * unit['mass']

# NPT integrator
from typing import Dict
def default_nhc_kwargs(tau: f64, overrides: Dict) -> Dict:
    base = {'chain_length': 3, 'chain_steps': 1, 'sy_steps': 1, 'tau': tau}
    return {**base, **(overrides or {})}

new_kwargs = {'chain_length': 3, 'chain_steps': 1, 'sy_steps': 1,}

init, apply = simulate.npt_nose_hoover(
    energy_fn=energy_full_jit,
    shift_fn=shift_fn_init,
    dt=dt,
    pressure=P_init,
    kT=T_init,
    barostat_kwargs=default_nhc_kwargs(tau_P, new_kwargs),
    thermostat_kwargs=default_nhc_kwargs(tau_T, new_kwargs)
)

state = init(key, positions, nlist=nlist, mass=mass, box=box)


# --- Logging setup ---
write_every = 100
log_steps = steps // write_every
log = {
    'PE': jnp.zeros((log_steps,)),
    'KE': jnp.zeros((log_steps,)),
    'kT': jnp.zeros((log_steps,)),
    'box': jnp.zeros((log_steps,) + box.shape)
}

# --- CPU-friendly batched stepping ---
batch_size = 5
num_batches = steps // batch_size

def make_run_batch(apply_fn, energy_fn, mass, batch_size, write_every):
    @jit
    def run_batch(state, nlist, global_step):
        def body(carry, i):
            state, nlist = carry
            state = apply_fn(state, nlist=nlist)
            nlist = neighbor_fn.update(state.position, nlist)

            mom = state.velocity * mass[:, None]
            kT = quantity.temperature(momentum=mom, mass=mass[:, None])
            KE = quantity.kinetic_energy(momentum=mom, mass=mass[:, None])

            do_pe = jnp.equal(jnp.mod(global_step + i, write_every), 0)
            PE = lax.cond(do_pe, lambda _: energy_fn(state.position, nlist, state.box), lambda _: jnp.nan, operand=None)

            obs = (kT, KE, PE, state.box)
            return (state, nlist), obs

        xs = jnp.arange(batch_size)
        (final_state, final_nlist), traj = lax.scan(body, (state, nlist), xs)
        return final_state, traj

    return run_batch

run_batch_jit = make_run_batch(apply, energy_full_jit, mass, batch_size, write_every)

global_step = 0
for batch in range(num_batches):
    state, traj = run_batch_jit(state, nlist, global_step)
    if (batch + 1) * batch_size % write_every == 0:
        log_idx = (batch + 1) * batch_size // write_every - 1
        log['kT'] = log['kT'].at[log_idx].set(traj[0][-1])
        log['KE'] = log['KE'].at[log_idx].set(traj[1][-1])
        log['PE'] = log['PE'].at[log_idx].set(traj[2][-1])
        log['box'] = log['box'].at[log_idx].set(state.box)
    global_step += batch_size

# --- Final energy ---
energy_final = energy_full_jit(state.position, nlist, state.box)
print(f"Total final potential: {energy_final:.6f} kcal/mol")

