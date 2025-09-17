import jax
import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad
from jax_md import units, quantity, space, partition, simulate
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from extract_params_oplsaa import parse_lammps_data
from modular_Ewald import PME_Coulomb, make_is_14_lookup
from typing import NamedTuple
import numpy as np
import pandas as pd
import os

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

displacement_fn, shift_fn = space.periodic(box)

# --- Neighbor list setup ---
neighbor_fn = partition.neighbor_list(
    displacement_fn, box, r_cutoff=cut_off_radius,
    dr_threshold=dR,
    mask=True,
    return_mask=True
)
nlist_active = neighbor_fn.allocate(positions)
nlist_next = neighbor_fn.allocate(positions)  # double buffer

n_atoms = positions.shape[0]
is_14_table = make_is_14_lookup(pair_indices, is_14_mask, n_atoms)

# Exclusions
exclusion_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)
bond_same_mol = molecule_id[bond_idx[:, 0]] == molecule_id[bond_idx[:, 1]]
angle_same_mol = molecule_id[angle_idx[:, 0]] == molecule_id[angle_idx[:, 2]]
bond_idx_filtered = bond_idx[bond_same_mol]
angle_idx_filtered = angle_idx[angle_same_mol]
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 0], bond_idx_filtered[:, 1]].set(True)
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 1], bond_idx_filtered[:, 0]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 0], angle_idx_filtered[:, 2]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 2], angle_idx_filtered[:, 0]].set(True)

# === Energy + Grad function ===
# === Coulomb Handler ===
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.16219451)

# === Build force field ===
bonded_lj_fn_factory_full, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id, box_size,
    use_soft_lj=False, exclusion_mask=exclusion_mask, is_14_table=is_14_table
)

def make_energy_and_grad(bonded_lj_factory, coulomb_handler,
                         charges, box_size, exclusion_mask, is_14_table):
    def energy_fn(R, nlist):
        # everything pure-JAX
        _, _, _, _, _, E_bonded_lj = bonded_lj_factory(R, nlist)
        _, _, _, E_coulomb = coulomb_handler.energy(
            R, charges, box_size, exclusion_mask, is_14_table, nlist
        )
        return E_bonded_lj + E_coulomb
    return jax.jit(jax.value_and_grad(energy_fn))

energy_grad_fn = make_energy_and_grad(
    bonded_lj_fn_factory_full, coulomb_handler,
    charges, box_size, exclusion_mask, is_14_table
)

print("Initial energy ...")
E_init, _ = energy_grad_fn(positions, nlist_active)
print(f"Total initial potential : {E_init:.6f} kcal/mol")

# --- NVT Langevin setup ---
timestep_fs = 1.0
dt = timestep_fs * unit['time']
tau_damp_fs = 200.0 * unit['time']
gamma = 1.0 / tau_damp_fs
write_every = 100
T_init = 298 * unit['temperature']
steps = 1000
key = random.PRNGKey(121)

mass = jnp.array(masses) * unit['mass']
mass_col = mass[:, None]

init, apply = simulate.nvt_langevin(
    lambda R, nlist: energy_grad_fn(R, nlist)[0],
    shift_fn,
    dt,
    T_init,
    gamma=gamma,
    mass=mass
)
state = init(key, positions, nlist=nlist_active)

# --- Device-side logging layout ---
n_frames = steps // write_every + 1

# Preallocate frame buffer on device
frames_init = jnp.zeros((n_frames,) + positions.shape, dtype=positions.dtype)

# Small counters to avoid modulo in the hot loop
nlist_update_steps = 10

# One step of the fused simulation.
def scan_step(carry, i):
    (state, nlist_act, nlist_nxt,
     since_update, since_write,
     frame_idx, frames) = carry

    # Thermo (device-side only)
    mom = state.velocity * mass_col
    T = quantity.temperature(momentum=mom, mass=mass_col)
    KE = quantity.kinetic_energy(momentum=mom, mass=mass_col)
    PE, _ = energy_grad_fn(state.position, nlist_act)

    # Update neighbor list into the "next" buffer when counter hits 0
    def do_nlist_update(nl_next):
        return nl_next.update(state.position)
    nlist_nxt = lax.cond(since_update == 0, do_nlist_update, lambda x: x, nlist_nxt)

    # Integrate one step using the ACTIVE nlist
    state = apply(state, nlist=nlist_act)

    # Swap buffers when we just rebuilt
    def do_swap(_):
        return (nlist_nxt, nlist_act)
    def no_swap(_):
        return (nlist_act, nlist_nxt)
    nlist_act, nlist_nxt = lax.cond(since_update == 0, do_swap, no_swap, operand=None)

    # Update the counters (wrap without %)
    since_update = lax.select(since_update == 0,
                              jnp.int32(nlist_update_steps - 1),
                              since_update - 1)
    since_write = lax.select(since_write == 0,
                             jnp.int32(write_every - 1),
                             since_write - 1)

    # Conditionally store a frame (device-side) when write counter hits 0
    def write_frame(frames_frame_idx):
        fr, idx = frames_frame_idx
        fr = fr.at[idx].set(state.position)
        return fr, idx + 1
    def skip_frame(frames_frame_idx):
        return frames_frame_idx

    frames, frame_idx = lax.cond(
        since_write == 0,
        write_frame,
        skip_frame,
        operand=(frames, frame_idx)
    )

    # Per-step outputs (kept on-device by scan)
    out = (T, KE, PE)

    carry = (state, nlist_act, nlist_nxt,
             since_update, since_write,
             frame_idx, frames)
    return carry, out

# Wrap the whole trajectory in one JIT
@jax.jit
def run_simulation(state, nlist_active, nlist_next):
    # Initialize counters so we write the initial frame and rebuild on step 0
    since_update0 = jnp.int32(0)
    since_write0  = jnp.int32(0)
    frame_idx0    = jnp.int32(0)
    frames0       = frames_init

    carry0 = (state, nlist_active, nlist_next,
              since_update0, since_write0,
              frame_idx0, frames0)

    # Run fused scan
    (state_f, nlist_act_f, nlist_nxt_f,
     _, _, frame_idx_f, frames_f), outs = lax.scan(
        scan_step, carry0, jnp.arange(steps)
    )

    # Stack per-step logs: outs = (T, KE, PE)
    T_traj, KE_traj, PE_traj = (outs[0], outs[1], outs[2])

    return (state_f, nlist_act_f, nlist_nxt_f, frame_idx_f, frames_f,
            T_traj, KE_traj, PE_traj)

print("Compiling & running fused trajectory ...")
(state, nlist_active, nlist_next, frame_idx, frames,
 T_traj, KE_traj, PE_traj) = run_simulation(state, nlist_active, nlist_next)

# Compute initial and final energies (optional; small extra work)
R_final = state.position
E_final, _ = energy_grad_fn(R_final, nlist_active)

# =========================
# 3) Single host transfer & I/O after the run
# =========================
trajectory_xyz_file = "trajectory.xyz"
thermo_file = "thermo.csv"
for f in [trajectory_xyz_file, thermo_file]:
    if os.path.exists(f):
        os.remove(f)

# Move data to host once
frames_np = np.array(frames[:int(frame_idx)])   # (n_written, N, 3)
kT_np     = np.array(T_traj)
KE_np     = np.array(KE_traj)
PE_np     = np.array(PE_traj)

# Save trajectory
with open(trajectory_xyz_file, "w") as f:
    for frame in frames_np:
        f.write(f"{frame.shape[0]}\n")
        f.write("Generated by JAX MD simulation\n")
        for atom_type, pos in zip(atom_types, frame):
            f.write(f"{atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

# Save thermo
import pandas as pd
df = pd.DataFrame({
    "step": np.arange(steps, dtype=int),
    "kT": kT_np,
    "KE": KE_np,
    "PE": PE_np,
})
df.to_csv(thermo_file, index=False)

print(f"Total final potential : {np.array(E_final):.6f} kcal/mol")
print(f"Trajectory saved to {trajectory_xyz_file}, thermo data saved to {thermo_file}")
