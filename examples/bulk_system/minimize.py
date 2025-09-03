#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jax import jit, grad, lax
from collections import namedtuple
from dataclasses import replace
from jax_md import space, partition
from jax_md.minimize import fire_descent
from extract_params_oplsaa import parse_lammps_data
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from modular_Ewald import PME_Coulomb, make_is_14_lookup
import jax.tree_util

# -------------------------
# Device selection (prefer GPU)
# -------------------------
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == "gpu"]
device = gpu_devices[0] if gpu_devices else devices[0]
print("Using device:", device)

# -------------------------
# Load system (user files)
# -------------------------
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses, atom_types  = parse_lammps_data(
    "EC_bulk.data", "EC.settings"
)
charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded
n_atoms = positions.shape[0]

# -------------------------
# Move key arrays to device
# -------------------------
positions = jax.device_put(positions, device=device)
charges = jax.device_put(charges, device=device)
box = jax.device_put(box, device=device)

# -------------------------
# Simulation parameters
# -------------------------
cut_off_radius = 15.0
dR = 0.5
force_tol = 1e-2
n_soft_steps = 100
n_full_steps = 2000
print_every = 50
neighbor_update_every = 10
n_log_steps = (n_soft_steps + n_full_steps) // print_every + 1

# -------------------------
# Periodic displacement & neighbor list
# -------------------------
displacement_fn, shift_fn = space.periodic(box)
neighbor_fn = partition.neighbor_list(
    displacement_fn, box, r_cutoff=cut_off_radius, dr_threshold=dR, mask=True, return_mask=True
)
nlist = neighbor_fn.allocate(positions)
nlist = jax.device_put(nlist, device=device)

# -------------------------
# Coulomb handler
# -------------------------
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.16219451)

# -------------------------
# Force field factories
# -------------------------
bonded_lj_fn_factory_soft, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers, nonbonded, molecule_id, box, use_soft_lj=True, lj_cap=1000.0
)
bonded_lj_fn_factory_full, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers, nonbonded, molecule_id, box, use_soft_lj=False
)

# -------------------------
# 1-4 table and exclusions
# -------------------------
is_14_table = make_is_14_lookup(pair_indices, is_14_mask, n_atoms)
is_14_table = jax.device_put(is_14_table, device=device)

exclusion_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)
bond_same_mol = molecule_id[bonds[0][:, 0]] == molecule_id[bonds[0][:, 1]]
angle_same_mol = molecule_id[angles[0][:, 0]] == molecule_id[angles[0][:, 2]]
bond_idx_filtered = bonds[0][bond_same_mol]
angle_idx_filtered = angles[0][angle_same_mol]
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 0], bond_idx_filtered[:, 1]].set(True)
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 1], bond_idx_filtered[:, 0]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 0], angle_idx_filtered[:, 2]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 2], angle_idx_filtered[:, 0]].set(True)
exclusion_mask = jax.device_put(exclusion_mask, device=device)

# -------------------------
# Energy (NO jit) and its gradient (outside jit)
# -------------------------
def energy_soft(R, nlist_local):
    E_bond = bonded_lj_fn_factory_soft(R, nlist_local)[-1]
    E_coul = coulomb_handler.energy(R, charges, box, exclusion_mask, is_14_table, nlist_local)[-1]
    return E_bond + E_coul

def energy_full(R, nlist_local):
    E_bond = bonded_lj_fn_factory_full(R, nlist_local)[-1]
    E_coul = coulomb_handler.energy(R, charges, box, exclusion_mask, is_14_table, nlist_local)[-1]
    return E_bond + E_coul

# Build grads ONCE, outside jit
grad_energy_soft = grad(energy_soft, argnums=0)
grad_energy_full = grad(energy_full, argnums=0)

# JIT small wrappers that call prebuilt grad
@jit
def energy_and_grad_soft(R, nlist_local):
    E = energy_soft(R, nlist_local)
    F = grad_energy_soft(R, nlist_local)
    return E, F

@jit
def energy_and_grad_full(R, nlist_local):
    E = energy_full(R, nlist_local)
    F = grad_energy_full(R, nlist_local)
    return E, F

# -------------------------
# Neighbor list update (JITed)
# -------------------------
update_nlist_jit = jit(lambda pos, nlist_local: neighbor_fn.update(pos, nlist_local))

# -------------------------
# FIRE optimizers
# -------------------------
energy_fn_soft = lambda R, **kwargs: energy_soft(R, kwargs["nlist"])
energy_fn_full = lambda R, **kwargs: energy_full(R, kwargs["nlist"])

init_soft, apply_soft = fire_descent(energy_fn_soft, shift_fn, dt_start=1e-6, f_inc=1.03)
init_full, apply_full = fire_descent(energy_fn_full, shift_fn, dt_start=1e-5, f_inc=1.01, dt_max=0.01)

opt_state_soft = init_soft(positions, nlist=nlist)
opt_state_soft = jax.tree_util.tree_map(lambda x: jax.device_put(x, device=device), opt_state_soft)

# -------------------------
# State
# -------------------------
MinState = namedtuple(
    "MinState",
    ["opt", "step_count", "energy", "max_force", "nlist", "log_idx", "energy_log", "force_log", "mode_full"],
)

state0 = MinState(
    opt=opt_state_soft,
    step_count=0,
    energy=0.0,
    max_force=jnp.inf,
    nlist=nlist,
    log_idx=0,
    energy_log=jnp.zeros(n_log_steps),
    force_log=jnp.zeros(n_log_steps),
    mode_full=False,
)

# -------------------------
# Step function (JITed)
# -------------------------
@jit
def step_fn(state):
    # 1) Neighbor list update (works with NeighborList via lax.cond)
    do_update_nlist = (state.step_count % neighbor_update_every == 0)
    nlist_next = lax.cond(
        do_update_nlist,
        lambda nl: update_nlist_jit(state.opt.position, nl),
        lambda nl: nl,
        operand=state.nlist,
    )

    # 2) Energy & gradient
    E_next, F_next = lax.cond(
        state.mode_full,
        lambda _: energy_and_grad_full(state.opt.position, nlist_next),
        lambda _: energy_and_grad_soft(state.opt.position, nlist_next),
        operand=None,
    )
    max_force = jnp.max(jnp.abs(F_next))

    # 3) FIRE update (branch inside lax.cond, returning PyTree)
    opt_next = lax.cond(
        state.mode_full,
        lambda _: apply_full(state.opt, nlist=nlist_next),
        lambda _: apply_soft(state.opt, nlist=nlist_next),
        operand=None,
    )

    # 4) Wrap positions in box
    opt_next = replace(opt_next, position=opt_next.position % box)

    # 5) Logging
    do_log = (state.step_count % print_every == 0)
    energy_log_next = lax.cond(
        do_log, lambda logs: logs.at[state.log_idx].set(E_next), lambda logs: logs, operand=state.energy_log
    )
    force_log_next = lax.cond(
        do_log, lambda logs: logs.at[state.log_idx].set(max_force), lambda logs: logs, operand=state.force_log
    )
    log_idx_next = lax.cond(do_log, lambda i: i + 1, lambda i: i, operand=state.log_idx)

    # 6) Switch to full mode after soft steps
    mode_full_next = jnp.where(state.step_count + 1 >= n_soft_steps, True, state.mode_full)

    return MinState(
        opt=opt_next,
        step_count=state.step_count + 1,
        energy=E_next,
        max_force=max_force,
        nlist=nlist_next,
        log_idx=log_idx_next,
        energy_log=energy_log_next,
        force_log=force_log_next,
        mode_full=mode_full_next,
    )

# -------------------------
# Loop condition
# -------------------------
def cond_fun(state):
    return jnp.logical_and(state.step_count < n_soft_steps + n_full_steps, state.max_force > force_tol)

# -------------------------
# Compile/warm-up (fast) and run
# -------------------------
# Warm up the small kernels first (much faster than compiling the whole while_loop immediately)
_ = energy_and_grad_soft(positions, nlist)
_ = energy_and_grad_full(positions, nlist)
_ = update_nlist_jit(positions, nlist)

# Warm up step_fn (single step compile)
_ = step_fn(state0)

# Run loop
state_final = lax.while_loop(cond_fun, step_fn, state0)
_ = state_final.opt.position.block_until_ready()

# -------------------------
# Results
# -------------------------
print(f"Final Energy: {float(state_final.energy):.6f}")
print(f"Final Max Force: {float(state_final.max_force):.6f}")
print(f"Total Steps Taken: {int(state_final.step_count)}")

for i in range(int(state_final.log_idx)):
    print(f"Step {i*print_every:5d}: Energy={float(state_final.energy_log[i]):.6f}, MaxForce={float(state_final.force_log[i]):.6f}")

