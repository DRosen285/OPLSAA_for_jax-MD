import jax
import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad
from jax_md import units, quantity, space, partition, simulate
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from extract_params_oplsaa import parse_lammps_data
from modular_Ewald import PME_Coulomb, make_is_14_lookup
from typing import NamedTuple, Dict
import numpy as np
import os
import time
from jax_md.util import f64

# === Load system ===
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses, atom_types = parse_lammps_data(
    'system_after_lammps_min.data',
    'EC.settings'
)

unit = units.real_unit_system()
charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded
bond_idx, k_b, r0 = bonds
angle_idx, k_theta, theta0 = angles
box_size = box
cut_off_radius = 15.0
dR = 0.5

# Use periodic_general if box may change in NPT (keeps flexibility)
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)

# --- Neighbor list setup ---
neighbor_fn = partition.neighbor_list(
    displacement_fn, box, r_cutoff=cut_off_radius,
    dr_threshold=dR,
    mask=True,
    return_mask=True
)
nlist_active = neighbor_fn.allocate(positions)  # initial neighbor list

n_atoms = positions.shape[0]
is_14_table = make_is_14_lookup(pair_indices, is_14_mask, n_atoms)

# Exclusions (same-molecule bonds/angles)
exclusion_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)
bond_same_mol = molecule_id[bond_idx[:, 0]] == molecule_id[bond_idx[:, 1]]
angle_same_mol = molecule_id[angle_idx[:, 0]] == molecule_id[angle_idx[:, 2]]
bond_idx_filtered = bond_idx[bond_same_mol]
angle_idx_filtered = angle_idx[angle_same_mol]
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 0], bond_idx_filtered[:, 1]].set(True)
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 1], bond_idx_filtered[:, 0]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 0], angle_idx_filtered[:, 2]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 2], angle_idx_filtered[:, 0]].set(True)

# === Build force field ===
bonded_lj_fn_factory_full, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id, box_size,
    use_soft_lj=False, exclusion_mask=exclusion_mask, is_14_table=is_14_table
)

# === Coulomb Handler ===
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.16219451)

# Energy function must accept box (for pressure/stress calculations)
def make_energy_fn(bonded_lj_factory, coulomb_handler,
                   charges, box_size, exclusion_mask, is_14_table):
    def energy_fn(R, nlist, box, **kwargs):
        # Note: many factories ignore box; if any part depends on box, pass it appropriately
        _, _, _, _, _, E_bonded_lj = bonded_lj_factory(R, nlist)
        _, _, _, E_coulomb = coulomb_handler.energy(
            R, charges, box_size, exclusion_mask, is_14_table, nlist
        )
        return E_bonded_lj + E_coulomb
    return jit(energy_fn)

energy_fn = make_energy_fn(
    bonded_lj_fn_factory_full, coulomb_handler,
    charges, box_size, exclusion_mask, is_14_table
)
energy_grad_fn = jit(value_and_grad(energy_fn))

# Quick initial energy check (host conversion for printing)
print("Initial energy ...")
E_init, _ = energy_grad_fn(positions, nlist_active, box)
E_init_val = float(jax.device_get(E_init))
print(f"Total initial potential : {E_init_val:.6f} kcal/mol")

# === NPT Nose-Hoover setup ===
npt_cycles = 5
npt_steps = 50
write_every = 50
nlist_update_steps = 10   # update neighbor list every N steps inside the inner loop

# --- NPT parameters ---
dt = 0.5 * unit['time']
tau_T = 200.0 * unit['time']
tau_P = 1000.0 * unit['time']
T_init = 298.0 * unit['temperature']
P_init = 1.0 * unit['pressure']
key = random.PRNGKey(121)

def default_nhc_kwargs(tau_physical: f64, overrides: Dict) -> Dict:
    # convert physical time constant to number of steps
    tau_in_steps = float(tau_physical / dt)
    base = {'chain_length': 3, 'chain_steps': 1, 'sy_steps': 1, 'tau': tau_in_steps}
    return {**base, **(overrides or {})}

new_kwargs = {'chain_length': 3, 'chain_steps': 1, 'sy_steps': 1}

mass = jnp.array(masses) * unit['mass']

# simulate.npt_nose_hoover: energy_fn must take (R, nlist, box)
init, apply = simulate.npt_nose_hoover(
    energy_fn,
    shift_fn=shift_fn,
    dt=dt,
    pressure=P_init,
    kT=T_init,
    barostat_kwargs=default_nhc_kwargs(tau_P, new_kwargs),
    thermostat_kwargs=default_nhc_kwargs(tau_T, new_kwargs)
)

# --- Convert arrays to NumPy for writing ---
n_atoms = positions.shape[0]
atom_ids = np.arange(1, n_atoms + 1)
atom_types_np = np.array(atom_types)
masses_np = np.array(masses)

# --- Open dump file (context manager) ---
dump_filename = "dump_EC.lammpstrj"
dump_file = open(dump_filename, "w")  # we'll close at the end; keep simple

# --- Initial state ---
# init expects key, positions, nlist, mass, box (some implementations vary; this matches your usage)
state = init(key, positions, nlist=nlist_active, mass=mass, box=box)
current_state = state
current_box = box
current_nbrs = nlist_active

# JIT the integrator apply function
apply = jax.jit(apply)

# Utility energy-only function for stress calc
def potential_energy_fn(R, nlist, box_arg):
    E, _ = energy_grad_fn(R, nlist, box_arg)
    return E
potential_energy_fn = jax.jit(potential_energy_fn)

# Build a per-step JIT body that advances integrator and conditionally updates neighborlist
def step_npt_body(i, carry):
    """
    i: integer step index (0..npt_steps-1)
    carry: (state, nbrs, box)
    """
    state, nbrs, box_here = carry
    # advance one NPT step
    state = apply(state, nlist=nbrs)

    # update neighbor list every nlist_update_steps using lax.cond
    do_update = jnp.equal(jnp.mod(i, nlist_update_steps), 0)
    # neighbor_fn.update is JAX-friendly and returns a new neighbor object
    def _update(_):
        # pass box explicitly if neighbor_fn supports it; otherwise it will ignore extra arg
        return neighbor_fn.update(state.position, nbrs)
    def _no_update(_):
        return nbrs
    nbrs_new = lax.cond(do_update, _update, _no_update, operand=None)

    return (state, nbrs_new, box_here)

step_npt = jax.jit(lambda state, nbrs, box_here: jax.lax.fori_loop(0, npt_steps, step_npt_body, (state, nbrs, box_here)))

# Diagnostics header
print('Step\tKE\tPE\tE_total\tT(K)\tP_avg(bar)\tTime/step(s)')
print('-----------------------------------------------------------------------------------')

# Track global step counter for dumps
global_step = 0
total_time_start = time.time()

for cycle in range(npt_cycles):
    old_time = time.time()

    # Run npt_steps in a single JIT fori_loop (device-side loop)
    new_state, new_nbrs, new_box = jax.block_until_ready(step_npt(current_state, current_nbrs, current_box))

    # Handle neighbor list overflow (host-side boolean check)
    overflow = False
    try:
        overflow = bool(jax.device_get(new_nbrs.did_buffer_overflow))
    except Exception:
        # some neighbor objects expose Python bool property
        overflow = bool(getattr(new_nbrs, 'did_buffer_overflow', False))

    if overflow:
        print("Neighbor list overflowed; reallocating on host.")
        # allocate with host arrays
        new_nbrs = neighbor_fn.allocate(jax.device_get(new_state.position), box=jax.device_get(new_box))

    # Accept state
    current_state = new_state
    current_nbrs = new_nbrs
    current_box = new_box

    # --- Energies & temperature (move only scalars to host for printing) ---
    KE = quantity.kinetic_energy(momentum=current_state.momentum, mass=current_state.mass)
    PE = potential_energy_fn(current_state.position, current_nbrs, current_box)
    T_inst = quantity.temperature(momentum=current_state.momentum, mass=current_state.mass) / unit['temperature']

    # --- Stress tensor + pressure ---
    # build a boxed energy wrapper for stress (quantity.stress expects energy(R, box, *args))
    def energy_for_stress(R, box, **kwargs):
        return energy_fn(R, nlist=current_nbrs, box=box, **kwargs)

    # quantity.stress returns virial-like 3x3
    sigma = quantity.stress(energy_for_stress, current_state.position, current_box)  # 3x3 virial
    # convert to pressure units: in jax-md quantity.stress often returns -virial; we follow your previous sign convention
    # P_virial_tensor has units of pressure after dividing by unit['pressure']
    P_virial_tensor = -sigma / unit['pressure']
    P_virial_scalar = jnp.trace(P_virial_tensor) / 3.0

    V = jnp.prod(current_box)
    # kinetic contribution to pressure (convert KE energy units to pressure units)
    P_kinetic = 2.0 / (3.0 * V) * (KE) / unit['pressure']  # dimensionless in units of 'pressure'

    P_tensor_total = P_virial_tensor + P_kinetic * jnp.eye(3)
    P_scalar_total = P_virial_scalar + P_kinetic

    # host conversions for printing
    KE_val = float(jax.device_get(KE))
    PE_val = float(jax.device_get(PE))
    E_tot_val = KE_val + PE_val
    T_val = float(jax.device_get(T_inst))
    P_val = float(jax.device_get(P_scalar_total))
    time_per_step = (time.time() - old_time) / float(npt_steps)

    global_step += npt_steps

    # Print & write every write_every cycles (global_step used for timesteps)
    if (cycle + 1) * npt_steps % write_every == 0:
        print(f"{global_step}\tKE={KE_val:.2f}\tPE={PE_val:.2f}\tE={E_tot_val:.3f}\t"
              f"T={T_val:.1f} K\tP_avg={P_val:.3f} bar\t"
              f"time/step={time_per_step:.4f}s")

        # --- LAMMPS-style dump ---
        f = dump_file
        f.write(f"ITEM: TIMESTEP\n{global_step}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
        f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
        box_arr = jax.device_get(current_box)
        box_arr = np.array(box_arr)

        # write orthorhombic bounds (if box stored as lengths)
        if box_arr.ndim == 1 and box_arr.size >= 3:
            f.write(f"0 {box_arr[0]}\n0 {box_arr[1]}\n0 {box_arr[2]}\n")
        elif box_arr.ndim == 2 and box_arr.shape[0] >= 3:
            # triclinic-ish: write diagonal lengths
            f.write(f"0 {box_arr[0,0]}\n0 {box_arr[1,1]}\n0 {box_arr[2,2]}\n")
        else:
            # fallback
            f.write(f"0 {box_arr.flatten()[0]}\n0 {box_arr.flatten()[1]}\n0 {box_arr.flatten()[2]}\n")

        f.write("ITEM: ATOMS id type mass x y z xu yu zu\n")
        pos_wrapped = jax.device_get(current_state.position)
        pos_unwrapped = pos_wrapped  # TODO: track images if you need true unwrapped coords

        for i in range(n_atoms):
            f.write(f"{i+1} {atom_types_np[i]} {masses_np[i]} "
                    f"{pos_wrapped[i,0]:.6f} {pos_wrapped[i,1]:.6f} {pos_wrapped[i,2]:.6f} "
                    f"{pos_unwrapped[i,0]:.6f} {pos_unwrapped[i,1]:.6f} {pos_unwrapped[i,2]:.6f}\n")

# --- Close dump file ---
dump_file.close()

print("Total simulation time: ", time.time() - total_time_start)
jax.clear_caches()

