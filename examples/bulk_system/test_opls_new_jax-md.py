import jax
import jax.numpy as jnp
from jax_md import space, partition
from types import SimpleNamespace

from jax_md.mm_forcefields import oplsaa
from jax_md.mm_forcefields.nonbonded.electrostatics import CutoffCoulomb
from jax_md.mm_forcefields.base import NonbondedOptions, Topology
from jax_md.mm_forcefields.oplsaa.params import Parameters
from jax_md.mm_forcefields import neighbor

from extract_params_oplsaa import parse_lammps_data

# === Load system from LAMMPS files ===
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses, atom_types = parse_lammps_data(
    'system_after_lammps_min.data',
    'EC.settings'
)

charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded
bond_idx, k_b, r0 = bonds
angle_idx, k_theta, theta0 = angles

n_atoms = positions.shape[0]

# === Neighbor list & displacement function ===
cut_off_radius = 15.0
dR = 0.5

displacement_fn, shift_fn = space.periodic(box)
neighbor_fn = partition.neighbor_list(displacement_fn, box, r_cutoff=cut_off_radius, dr_threshold=dR, mask=True)
nlist_init = neighbor_fn.allocate(positions)

# === Exclusions and 1-4 mask ===
#build exclusion list for non-bonded interactions
exclusion_mask = neighbor.make_exclusion_mask(
    n_atoms,
    bond_idx,      # shape [n_bonds, 2]
    angle_idx,     # shape [n_angles, 3]
    molecule_id    # shape [n_atoms]
)

# Build 1-4 interaction table for scaling

pair_14_mask = neighbor.make_14_table(
    n_atoms,
    torsions[0],       # torsions[0] should be torsion indices, shape [n_torsions, 4]
    exclusion_mask,
    molecule_id
)

# === Topology ===
topo = Topology(
    n_atoms=n_atoms,
    bonds=jnp.array(bonds[0], dtype=int) if bonds is not None else jnp.zeros((0,2), dtype=int),
    angles=jnp.array(angles[0], dtype=int) if angles is not None else jnp.zeros((0,3), dtype=int),
    torsions=jnp.array(torsions[0], dtype=int) if torsions is not None else jnp.zeros((0,4), dtype=int),
    impropers=jnp.array(impropers[0], dtype=int) if impropers is not None else jnp.zeros((0,4), dtype=int),
    exclusion_mask=exclusion_mask,
    pair_14_mask=pair_14_mask
)

# === Parameters using SimpleNamespace ===
bonded_params = SimpleNamespace(
    bond_k = jnp.array(bonds[1]) if bonds is not None else jnp.array([]),
    bond_r0 = jnp.array(bonds[2]) if bonds is not None else jnp.array([]),
    angle_k = jnp.array(angles[1]) if angles is not None else jnp.array([]),
    angle_theta0 = jnp.array(angles[2]) if angles is not None else jnp.array([]),
    torsion_k = jnp.array(torsions[1]) if torsions is not None else jnp.array([]),
    torsion_n = jnp.array(torsions[3]) if torsions is not None else jnp.array([]),
    torsion_gamma = jnp.array(torsions[4]) if torsions is not None else jnp.array([]),
    improper_k = jnp.array(impropers[1]) if impropers is not None else jnp.array([]),
    improper_n = jnp.array(impropers[3]) if impropers is not None else jnp.array([]),
    improper_gamma = jnp.array(impropers[4]) if impropers is not None else jnp.array([])
)

nonbonded_params = SimpleNamespace(
    charges = charges,
    sigma   = sigmas,
    epsilon = epsilons
)

params = Parameters(
    bonded = bonded_params,
    nonbonded = nonbonded_params
)

# === Coulomb handler and nonbonded options ===
coulomb_handler = CutoffCoulomb(r_cut=cut_off_radius)
nb_options = NonbondedOptions(r_cut=cut_off_radius, dr_threshold=dR)

# === Create energy function ===
energy_fn, neighbor_fn, disp_fn = oplsaa.energy(topo, params, box, coulomb_handler, nb_options)

# === Compute energies ===
E_dict = energy_fn(positions, nlist_init)

# === Print breakdown ===
print("\nEnergy terms (kcal/mol):")
for key, val in E_dict.items():
    print(f"{key:12s}: {val:.6f}")

