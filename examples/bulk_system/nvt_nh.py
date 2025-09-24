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
import time

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

# === Build force field ===
bonded_lj_fn_factory_full, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id, box_size,
    use_soft_lj=False,exclusion_mask=exclusion_mask, is_14_table=is_14_table
)

# === Coulomb Handler ===
coulomb_handler = PME_Coulomb(grid_size=32, alpha=0.16219451)

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

# --- Build topology object for trajectory saving/visualization ---
import mdtraj as md
def build_topology(atom_types, molecule_id, bond_idx):
    """
    Build MDTraj Topology from parsed system info.
    """
    top = md.Topology()
    chains = {}

    # Ensure molecule_id is a numpy array of ints
    molecule_id = np.array(molecule_id).astype(int)
    atom_types = np.array(atom_types)  # optional

    for mol_id in np.unique(molecule_id):
        chain = top.add_chain()
        res = top.add_residue(f"MOL{mol_id}", chain)
        chains[mol_id] = res

    atoms = []
    for i, (atype, mol_id) in enumerate(zip(atom_types, molecule_id)):
        mol_id = int(mol_id)  # <-- convert to native Python int
        # map atom type to MDTraj element if possible
        if isinstance(atype, str) and len(atype) <= 2:
            elem = md.element.get_by_symbol(atype)
        else:
            elem = md.element.carbon  # fallback
        atom = top.add_atom(f"{atype}{i}", elem, chains[mol_id])
        atoms.append(atom)

    # add bonds
    for i, j in bond_idx:
        atom_i = int(i)
        atom_j = int(j)
        top.add_bond(atoms[atom_i], atoms[atom_j])

    return top



# === NVT Nose-Hoover setup ===
nvt_cycles = 5
nvt_steps = 50
write_every = 10  # match LAMMPS dump frequency

timestep_fs = 1.0
dt = timestep_fs * unit['time']
tau_T=100.0 * unit['time']
write_every = 100
T_init = 298 * unit['temperature']
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

# --- Convert arrays to NumPy for writing ---
n_atoms = positions.shape[0]
atom_ids = np.arange(1, n_atoms + 1)
atom_types_np = np.array(atom_types)
masses_np = np.array(masses)

# --- Open dump file ---
dump_filename = "dump_EC.lammpstrj"
dump_file = open(dump_filename, "w")


# --- Initial state ---
state = init(key, positions, nlist=nlist_active)
current_state = state
current_box = box
current_nbrs = nlist_active

# JIT the integrator functions for speed
apply = jax.jit(apply)
init = jax.jit(init)

# Initial neighbor object
nbrs = nlist_active

# --- Energy/kinetic functions ---
kinetic_energy = quantity.kinetic_energy
temperature_fn = quantity.temperature

def potential_energy_fn(R, nlist):
    E, _ = energy_grad_fn(R, nlist)
    return E
potential_energy_fn = jax.jit(potential_energy_fn)

# --- Step function for JAX ---
@jax.jit
def step_nvt_fn(i, carry):
    state, nbrs, box, kT = carry

    state = apply(state, nlist=nbrs)

    # Update neighbor list (purely functional)
    nbrs = neighbor_fn.update(state.position, nbrs)

    return (state, nbrs, box, kT)


# --- Main simulation loop ---
total_time_start = time.time()
print('Step\tKE\tPE\tTotal Energy\tTemperature\ttime/step (s)')
print('-----------------------------------------------------------------------------------')

for cycle in range(nvt_cycles):
    temp_i = T_init
    old_time = time.time()

    # Run NVT steps in JIT loop
    carry_init = (current_state, current_nbrs, current_box, temp_i)
    new_state, new_nbrs, new_box, _ = jax.block_until_ready(
        jax.lax.fori_loop(0, nvt_steps, step_nvt_fn, carry_init)
    )

    # Check neighbor overflow
    if new_nbrs.did_buffer_overflow:
        print("Neighbor list overflowed, reallocating.")
        new_nbrs = neighbor_fn.allocate(new_state.position, box=new_box)

    # Accept new state
    current_state = new_state
    current_nbrs = new_nbrs
    current_box = new_box

    # --- Diagnostics ---
    KE = kinetic_energy(momentum=current_state.momentum, mass=current_state.mass)
    PE = potential_energy_fn(current_state.position, current_nbrs)
    T_inst = temperature_fn(momentum=current_state.momentum, mass=current_state.mass) / unit['temperature']

    steps_done = cycle * nvt_steps
    time_per_step = (time.time() - old_time) / nvt_steps

    print(f"{steps_done}\t{KE:.2f}\t{PE:.2f}\t{(KE+PE):.3f}\t{T_inst:.1f}\t{time_per_step:.4f}")

    # --- LAMMPS-style dump ---
    if steps_done % write_every == 0:
        f = dump_file
        f.write(f"ITEM: TIMESTEP\n{steps_done}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
        f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
        # Handle both 1D (orthorhombic) and 2D (triclinic) cases
        box_arr = np.array(current_box)

        if box_arr.ndim == 1:
        # Orthorhombic: [Lx, Ly, Lz]
          f.write(f"0 {box_arr[0]}\n")
          f.write(f"0 {box_arr[1]}\n")
          f.write(f"0 {box_arr[2]}\n")
        elif box_arr.ndim == 2:
        # Triclinic: use diagonal as lengths
          f.write(f"0 {box_arr[0,0]}\n")
          f.write(f"0 {box_arr[1,1]}\n")
          f.write(f"0 {box_arr[2,2]}\n")
        else:
          raise ValueError(f"Unexpected box shape: {box_arr.shape}")


        f.write("ITEM: ATOMS id type mass x y z xu yu zu\n")

        pos_wrapped = np.array(current_state.position)
        pos_unwrapped = pos_wrapped  # replace with actual unwrapped if available

        for i in range(n_atoms):
            f.write(f"{atom_ids[i]} {atom_types_np[i]} {masses_np[i]} "
                    f"{pos_wrapped[i,0]:.6f} {pos_wrapped[i,1]:.6f} {pos_wrapped[i,2]:.6f} "
                    f"{pos_unwrapped[i,0]:.6f} {pos_unwrapped[i,1]:.6f} {pos_unwrapped[i,2]:.6f}\n")

# --- Close dump file ---
dump_file.close()
print("Total simulation time: ", time.time() - total_time_start)
jax.clear_caches()
