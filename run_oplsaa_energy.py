import numpy as np
import jax.numpy as jnp
from energy_oplsaa import opls_aa_energy

def parse_lammps_data(filepath):
    from collections import defaultdict

    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None
    data = defaultdict(list)

    for line in lines:
        if 'Masses' in line:
            section = 'Masses'; continue
        elif 'Pair Coeffs' in line:
            section = 'Pair Coeffs'; continue
        elif 'Bond Coeffs' in line:
            section = 'Bond Coeffs'; continue
        elif 'Angle Coeffs' in line:
            section = 'Angle Coeffs'; continue
        elif 'Dihedral Coeffs' in line:
            section = 'Dihedral Coeffs'; continue
        elif 'Improper Coeffs' in line:
            section = 'Improper Coeffs'; continue
        elif 'Atoms' in line:
            section = 'Atoms'; continue
        elif 'Bonds' in line:
            section = 'Bonds'; continue
        elif 'Angles' in line:
            section = 'Angles'; continue
        elif 'Dihedrals' in line:
            section = 'Dihedrals'; continue
        elif 'Impropers' in line:
            section = 'Impropers'; continue
        elif not line.strip() or line.startswith('#'):
            continue
        data[section].append(line.strip())

    num_atoms = len(data['Atoms'])
    positions = np.zeros((num_atoms, 3))
    charges = np.zeros((num_atoms,))
    types = np.zeros((num_atoms,), dtype=int)

    for i, line in enumerate(data['Atoms']):
        parts = line.split()
        idx, mol, typ, q, x, y, z = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        positions[i] = [x, y, z]
        charges[i] = q
        types[i] = typ

    pair_lines = sorted(data['Pair Coeffs'], key=lambda x: int(x.split()[0]))
    epsilons_by_type = np.array([float(l.split()[1]) for l in pair_lines])
    sigmas_by_type = np.array([float(l.split()[2]) for l in pair_lines])
    sigmas = sigmas_by_type[types - 1]
    epsilons = epsilons_by_type[types - 1]

    bond_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1] for l in data['Bonds']])
    bond_type = np.array([int(l.split()[1]) - 1 for l in data['Bonds']])
    bond_params = np.array([[float(x.split()[1]), float(x.split()[2])] for x in sorted(data['Bond Coeffs'], key=lambda x: int(x.split()[0]))])
    k_b = bond_params[bond_type, 0]
    r0 = bond_params[bond_type, 1]

    angle_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1, int(l.split()[4]) - 1] for l in data['Angles']])
    angle_type = np.array([int(l.split()[1]) - 1 for l in data['Angles']])
    angle_params = np.array([[float(x.split()[1]), float(x.split()[2])] for x in sorted(data['Angle Coeffs'], key=lambda x: int(x.split()[0]))])
    k_theta = angle_params[angle_type, 0]
    theta0 = np.deg2rad(angle_params[angle_type, 1])

    torsion_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1, int(l.split()[4]) - 1, int(l.split()[5]) - 1] for l in data['Dihedrals']])
    # Map dihedral to its type (zero-based)
    torsion_type = np.array([int(l.split()[1]) - 1 for l in data['Dihedrals']])
    # Parse and sort dihedral coefficient lines
    torsion_raw = [x.split() for x in sorted(data['Dihedral Coeffs'], key=lambda x: int(x.split()[0]))]
    # Initialize parameter arrays
    num_torsions = len(torsion_idx)
    k_torsion = np.zeros(num_torsions)
    n_torsion = np.zeros(num_torsions)
    d_torsion = np.zeros(num_torsions)
    gamma_torsion = np.zeros(num_torsions)
    # Populate K, d, n from the dihedral type lookup
    for i, t in enumerate(torsion_type):
        coeffs = torsion_raw[t]
        k_torsion[i] = float(coeffs[1])       # force constant
        d_torsion[i] = int(coeffs[2])         # d: +1 or -1 n
        n_torsion[i] = float(coeffs[3])      # integer
        gamma_torsion[i] = 0.0 if d_torsion[i] == 1 else jnp.pi


    improper_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1, int(l.split()[4]) - 1, int(l.split()[5]) - 1] for l in data['Impropers']])
    improper_type = np.array([int(l.split()[1]) - 1 for l in data['Impropers']])
    #improper_params = np.array([[float(x.split()[1]), float(x.split()[2])] for x in sorted(data['Improper Coeffs'], key=lambda x: int(x.split()[0]))])
    improper_params = [x.split() for x in sorted(data['Improper Coeffs'], key=lambda x: int(x.split()[0]))]
    #k_psi = improper_params[improper_type, 0]
    #psi0 = improper_params[improper_type, 1]
        # Initialize parameter arrays
    num_improper = len(improper_idx)
    k_improper = np.zeros(num_improper)
    n_improper = np.zeros(num_improper)
    d_improper = np.zeros(num_improper)
    gamma_improper = np.zeros(num_improper)
    for i, t in enumerate(improper_type):
        coeffs = improper_params[t]
        k_improper[i] = float(coeffs[1])       # force constant
        d_improper[i] = int(coeffs[2])         # d: +1 or -1 n
        n_improper[i] = float(coeffs[3])      # integer
        gamma_improper[i] = 0.0 if d_improper[i] == 1 else jnp.pi

    def get_nonbonded_pairs_with_14_mask(num_atoms, bond_idx, angle_idx, torsion_idx, improper_idx):
    #"""
    #Computes all nonbonded atom pairs (i < j), excluding 1–2 (bonds) and 1–3 (angles),
    #but keeping 1–4 interactions (from torsions and impropers).
    #Returns a boolean mask for which nonbonded pairs are 1–4 and should be scaled.

    #Args:
    #    num_atoms (int)
    #    bond_idx: shape (Nb, 2)
    #    angle_idx: shape (Na, 3)
    #    torsion_idx: shape (Nt, 4)
    #    improper_idx: shape (Ni, 4)

    #Returns:
    #    nonbonded_pairs: (Nn, 2)
    #    is_14_mask: (Nn,) boolean array where True = 1–4 interaction
    #"""

    # Step 1: All possible i < j atom pairs
        i_vals, j_vals = jnp.triu_indices(num_atoms, k=1)
        pair_indices = jnp.stack([i_vals, j_vals], axis=1)
        pair_indices = jnp.sort(pair_indices, axis=1)

    # Step 2: Exclude 1–2 (bonds) and 1–3 (angles)
        excluded_pairs = jnp.concatenate([
           jnp.sort(bond_idx[:, [0, 1]], axis=1),
           jnp.sort(angle_idx[:, [0, 2]], axis=1),
           ], axis=0)
        excluded_pairs = jnp.unique(excluded_pairs, axis=0)

    # Step 3: Compute 1–4 pairs from torsions:
        one_four_pairs = jnp.sort(torsion_idx[:, [0, 3]], axis=1)
        one_four_pairs = jnp.unique(one_four_pairs, axis=0)

    # Step 4: Remove excluded pairs from all
        pair_indices_exp = pair_indices[:, None, :]
        excluded_pairs_exp = excluded_pairs[None, :, :]
        is_excluded = jnp.any(jnp.all(pair_indices_exp == excluded_pairs_exp, axis=-1), axis=1)
        nonbonded_pairs = pair_indices[~is_excluded]

    # Step 5: Mark which of those are 1–4 (for scaling)
        nb_exp = nonbonded_pairs[:, None, :]
        one_four_exp = one_four_pairs[None, :, :]
        is_14_mask = jnp.any(jnp.all(nb_exp == one_four_exp, axis=-1), axis=1)

        return nonbonded_pairs, is_14_mask




    pair_indices, is_14_mask = get_nonbonded_pairs_with_14_mask(num_atoms, bond_idx, angle_idx, torsion_idx, improper_idx)
    #np.array([[i, j] for i in range(num_atoms) for j in range(i + 1, num_atoms)])
    box = jnp.array([54.0, 54.0, 54.0])

    return (
        jnp.array(positions), 
        (jnp.array(bond_idx), jnp.array(k_b), jnp.array(r0)),
        (jnp.array(angle_idx), jnp.array(k_theta), jnp.array(theta0)),
        (jnp.array(torsion_idx), jnp.array(k_torsion),jnp.array(d_torsion),jnp.array(n_torsion),jnp.array(gamma_torsion)),
        (jnp.array(improper_idx), jnp.array(k_improper), jnp.array(d_improper),jnp.array(n_improper),jnp.array(gamma_improper)),
        (jnp.array(charges), jnp.array(sigmas), jnp.array(epsilons), jnp.array(pair_indices),is_14_mask),
        box
    )
