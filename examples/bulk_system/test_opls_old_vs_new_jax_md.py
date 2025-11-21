import jax
import jax.numpy as jnp
from jax_md import space, partition
from types import SimpleNamespace

from jax_md.mm_forcefields import oplsaa
from jax_md.mm_forcefields.nonbonded.electrostatics import CutoffCoulomb,PMECoulomb,EwaldCoulomb
from jax_md.mm_forcefields.base import NonbondedOptions, Topology
from jax_md.mm_forcefields.oplsaa.params import Parameters

from extract_params_oplsaa import parse_lammps_data
from modular_Ewald import make_is_14_lookup
from energy_oplsaa import optimized_opls_aa_energy_with_nlist_modular
from modular_Ewald import CutoffCoulomb as CutoffCoulomb_old 
from modular_Ewald import PME_Coulomb as PME_Coulomb_old
from modular_Ewald import EwaldCoulomb as EwaldCoulomb_old
# === Load system ===
positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses, atom_types = parse_lammps_data(
    'system_after_lammps_min.data',
    'EC.settings'
)

charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded
bond_idx, k_b, r0 = bonds
angle_idx, k_theta, theta0 = angles

n_atoms = positions.shape[0]

# === Neighbor list & displacement ===
cut_off_radius = 15.0
dR = 0.5
displacement_fn, shift_fn = space.periodic(box)
neighbor_fn = partition.neighbor_list(displacement_fn, box, r_cutoff=cut_off_radius, dr_threshold=dR, mask=True)
nlist_init = neighbor_fn.allocate(positions)

# === Exclusions & 1-4 mask ===
exclusion_mask = jnp.zeros((n_atoms, n_atoms), dtype=bool)
pair_14_mask = make_is_14_lookup(pair_indices, is_14_mask, n_atoms)

bond_same_mol = molecule_id[bond_idx[:, 0]] == molecule_id[bond_idx[:, 1]]
angle_same_mol = molecule_id[angle_idx[:, 0]] == molecule_id[angle_idx[:, 2]]

bond_idx_filtered = bond_idx[bond_same_mol]
angle_idx_filtered = angle_idx[angle_same_mol]

exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 0], bond_idx_filtered[:, 1]].set(True)
exclusion_mask = exclusion_mask.at[bond_idx_filtered[:, 1], bond_idx_filtered[:, 0]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 0], angle_idx_filtered[:, 2]].set(True)
exclusion_mask = exclusion_mask.at[angle_idx_filtered[:, 2], angle_idx_filtered[:, 0]].set(True)

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

# === Parameters (raw, no conversion yet) ===
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


# === Coulomb handler and options ===
coulomb_handler = CutoffCoulomb(r_cut=cut_off_radius)
coulomb_handler = PMECoulomb(grid_size=32, alpha=0.16219451)
coulomb_handler = EwaldCoulomb(alpha=0.1540129, kmax=5, r_cut=cut_off_radius)

#coulomb_handler_old=CutoffCoulomb_old(r_cut=cut_off_radius)
#coulomb_handler_old = PME_Coulomb_old(grid_size=32, alpha=0.16219451)
coulomb_handler_old = EwaldCoulomb_old(alpha=0.1540129, kmax=5, r_cut=cut_off_radius)

nb_options = NonbondedOptions(r_cut=cut_off_radius, dr_threshold=dR)

# === New implementation energy function ===
energy_fn_new, _, _ = oplsaa.energy(topo, params, box, coulomb_handler, nb_options)

# === Old implementation factory ===
bonded_lj_fn_factory_old, _, _ = optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers, nonbonded, molecule_id, box,
    use_soft_lj=False
)

# === Full breakdown function for old implementation ===
def energy_breakdown_old(R, nlist, coulomb_handler):
    # Bonded + nonbonded energies
    E_bond, E_angle, E_torsion, E_improper, E_nb, E_bonded_lj = bonded_lj_fn_factory_old(R, nlist)

    # Coulomb contributions
    e_real, e_recip, e_self, E_coulomb = coulomb_handler.energy(
        R, charges, box, exclusion_mask, pair_14_mask, nlist
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
        E_total
    )

# === Compute energies ===
E_old_breakdown = energy_breakdown_old(positions, nlist_init, coulomb_handler_old)
E_new_breakdown = energy_fn_new(positions, nlist_init)

# === Print per-term comparison (dynamic old values) ===
terms = ['bond', 'angle', 'torsion', 'improper', 'lj']
print("\nPer-term comparison (kcal/mol):")
print(f"{'Term':12s}{'Old':>12s}{'New':>12s}{'Diff':>12s}")

for i, term in enumerate(terms):
    old_val = E_old_breakdown[i].item() if hasattr(E_old_breakdown[i], 'item') else E_old_breakdown[i]
    new_val = E_new_breakdown[term].item()
    diff = new_val - old_val
    print(f"{term:12s}{old_val:12.6f}{new_val:12.6f}{diff:12.6f}")

# Total energy
E_old_total = E_old_breakdown[-1].item() if hasattr(E_old_breakdown[-1], 'item') else E_old_breakdown[-1]
E_new_total = E_new_breakdown['total'].item()
print(f"\n{'Total':12s}{E_old_total:12.6f}{E_new_total:12.6f}{E_new_total - E_old_total:12.6f}")
# === Coulomb comparison ===
(
    E_bond_old,
    E_angle_old,
    E_torsion_old,
    E_improper_old,
    E_lj_old,
    e_real_old,
    e_recip_old,
    e_self_old,
    E_coulomb_old,
    E_total_old
) = E_old_breakdown

print("\nCoulomb comparison (kcal/mol):")
print(f"{'Term':12s}{'Old':>12s}{'New':>12s}{'Diff':>12s}")

# New implementation currently provides only total Coulomb = new['coulomb']
E_coulomb_new = E_new_breakdown["coulomb"].item()

# If you want only Coulomb total:
print(f"{'coulomb':12s}{E_coulomb_old:12.6f}{E_coulomb_new:12.6f}{E_coulomb_new - E_coulomb_old:12.6f}")

# Optional: print Ewald sub-terms if desired (old impl only)
print(f"{'real':12s}{e_real_old:12.6f}{'   n/a':>12s}{'   n/a':>12s}")
print(f"{'recip':12s}{e_recip_old:12.6f}{'   n/a':>12s}{'   n/a':>12s}")
print(f"{'self':12s}{e_self_old:12.6f}{'   n/a':>12s}{'   n/a':>12s}")

