import jax
import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad
from jax_md import units, quantity, space, partition, simulate
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from extract_params_oplsaa import parse_lammps_data
from modular_Ewald import PME_Coulomb, make_is_14_lookup
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
        _, _, _, _, _, E_bonded_lj = bonded_lj_factory(R, nlist)
        _, _, _, E_coulomb = coulomb_handler.energy(
            R, charges, box_size, exclusion_mask, is_14_table, nlist
        )
        return E_bonded_lj + E_coulomb
    return jit(value_and_grad(energy_fn))

energy_grad_fn = make_energy_and_grad(
    bonded_lj_fn_factory_full, coulomb_handler,
    charges, box_size, exclusion_mask, is_14_table
)

print("Initial energy ...")
E_init, _ = energy_grad_fn(positions, nlist_active)
print(f"Total initial potential : {E_init:.6f} kcal/mol")

# === NVT Nose-Hoover setup ===
timestep_fs = 1.0
dt = timestep_fs * unit['time']
tau_T=100.0 * unit['time']
write_every = 100
T_init = 298 * unit['temperature']
steps = 1000
key = random.PRNGKey(121)

mass = jnp.array(masses) * unit['mass']
mass_col = mass[:, None]

init, apply =  simulate.nvt_nose_hoover(
    lambda R, nlist: energy_grad_fn(R, nlist)[0],
    shift_fn,
    dt=dt,
    kT=T_init,
    tau=tau_T,
    mass=mass
)
state = init(key, positions, nlist=nlist_active)

# --- Simulation step for scan ---
nlist_update_steps = 10

def scan_step(carry, _):
    state, nlist_active, nlist_next, counter = carry

    # Thermo quantities
    mom = state.velocity * mass_col
    T = quantity.temperature(momentum=mom, mass=mass_col)
    KE = quantity.kinetic_energy(momentum=mom, mass=mass_col)
    PE, _ = energy_grad_fn(state.position, nlist_active)

    # Neighbor list update
    nlist_next = lax.cond(
        counter == 0,
        lambda nl: nl.update(state.position),
        lambda nl: nl,
        nlist_next
    )
    state = apply(state, nlist=nlist_active)

    # Swap lists if updated
    nlist_active, nlist_next, counter = lax.cond(
        counter == 0,
        lambda _: (nlist_next, nlist_active, nlist_update_steps),
        lambda _: (nlist_active, nlist_next, counter - 1),
        operand=None
    )

    carry = (state, nlist_active, nlist_next, counter)
    logs = (T, KE, PE, state.position)
    return carry, logs

# --- Run scan ---
carry_init = (state, nlist_active, nlist_next, nlist_update_steps)
(carry_final, logs) = lax.scan(scan_step, carry_init, None, length=steps)

final_state, nlist_active, nlist_next, _ = carry_final
kT_log, KE_log, PE_log, positions_log = logs

# --- Apply thinning ---
write_mask = jnp.arange(steps) % write_every == 0
kT_out = np.array(kT_log[write_mask])
KE_out = np.array(KE_log[write_mask])
PE_out = np.array(PE_log[write_mask])
positions_out = np.array(positions_log[write_mask])

# === Saving ===
trajectory_xyz_file = "trajectory.xyz"
thermo_file = "thermo.csv"

for f in [trajectory_xyz_file, thermo_file]:
    if os.path.exists(f):
        os.remove(f)

def save_xyz(positions_out, atom_types, file_path):
    with open(file_path, "a") as f:
        for frame in positions_out:
            f.write(f"{len(atom_types)}\n")
            f.write("Generated by JAX MD simulation\n")
            for atom_type, pos in zip(atom_types, frame):
                f.write(f"{atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

def save_thermo(kT_out, KE_out, PE_out, write_every, thermo_file):
    steps_out = np.arange(0, steps, write_every)
    df = pd.DataFrame({
        "step": steps_out,
        "kT": kT_out,
        "KE": KE_out,
        "PE": PE_out
    })
    df.to_csv(thermo_file, index=False)

# Save all at once
save_xyz(positions_out, atom_types, trajectory_xyz_file)
save_thermo(kT_out, KE_out, PE_out, write_every, thermo_file)

# === Final energy ===
R_final = final_state.position
E_final, _ = energy_grad_fn(R_final, nlist_active)
print(f"Total final potential : {E_final:.6f} kcal/mol")
print(f"Trajectory saved to {trajectory_xyz_file}, thermo data saved to {thermo_file}")

