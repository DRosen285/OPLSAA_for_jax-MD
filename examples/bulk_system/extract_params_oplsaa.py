import numpy as np
import jax.numpy as jnp
from collections import defaultdict

def parse_force_field_settings(settings_path):
    with open(settings_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    ff_data = defaultdict(list)
    for line in lines:
        parts = line.split()
        key = parts[0]
        ff_data[key].append(parts[1:])

    pair_coeffs = sorted(ff_data['pair_coeff'], key=lambda x: int(x[0]))
    epsilons_by_type = np.array([float(x[2]) for x in pair_coeffs])
    sigmas_by_type = np.array([float(x[3]) for x in pair_coeffs])

    bond_coeffs = sorted(ff_data['bond_coeff'], key=lambda x: int(x[0]))
    bond_params = np.array([[float(x[1]), float(x[2])] for x in bond_coeffs])

    angle_coeffs = sorted(ff_data['angle_coeff'], key=lambda x: int(x[0]))
    angle_params = np.array([[float(x[1]), float(x[2])] for x in angle_coeffs])

    dihedral_coeffs = sorted(ff_data['dihedral_coeff'], key=lambda x: int(x[0]))
    torsion_params = [[float(x[1]), int(x[2]), float(x[3])] for x in dihedral_coeffs]

    improper_coeffs = sorted(ff_data['improper_coeff'], key=lambda x: int(x[0]))
    improper_params = [[float(x[1]), int(x[2]), float(x[3])] for x in improper_coeffs]

    return epsilons_by_type, sigmas_by_type, bond_params, angle_params, torsion_params, improper_params
def parse_lammps_data(data_path, settings_path):
    eps_by_type, sig_by_type, bond_params, angle_params, torsion_raw, improper_raw = parse_force_field_settings(settings_path)

    with open(data_path, 'r') as f:
        lines = f.readlines()

    section = None
    data = defaultdict(list)

    for line in lines:
        if 'Masses' in line:
            section = 'Masses'; continue
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
    molecule_id = np.zeros((num_atoms,))

    for i, line in enumerate(data['Atoms']):
        parts = line.split()
        idx, mol, typ, q, x, y, z = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        positions[i] = [x, y, z]
        charges[i] = q
        types[i] = typ
        molecule_id[i] = mol

    # 🔹 Parse masses and map to atoms
    mass_lines = data['Masses']
    num_types = max(types)
    masses_by_type = np.zeros(num_types)
    for line in mass_lines:
        parts = line.split()
        typ = int(parts[0])
        mass = float(parts[1])
        masses_by_type[typ - 1] = mass
    masses = masses_by_type[types - 1]

    sigmas = sig_by_type[types - 1]
    epsilons = eps_by_type[types - 1]

    bond_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1] for l in data['Bonds']])
    bond_type = np.array([int(l.split()[1]) - 1 for l in data['Bonds']])
    k_b = bond_params[bond_type, 0]
    r0 = bond_params[bond_type, 1]

    angle_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1, int(l.split()[4]) - 1] for l in data['Angles']])
    angle_type = np.array([int(l.split()[1]) - 1 for l in data['Angles']])
    k_theta = angle_params[angle_type, 0]
    theta0 = np.deg2rad(angle_params[angle_type, 1])

    torsion_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1, int(l.split()[4]) - 1, int(l.split()[5]) - 1] for l in data['Dihedrals']])
    torsion_type = np.array([int(l.split()[1]) - 1 for l in data['Dihedrals']])
    k_torsion = np.array([torsion_raw[t][0] for t in torsion_type])
    d_torsion = np.array([torsion_raw[t][1] for t in torsion_type])
    n_torsion = np.array([torsion_raw[t][2] for t in torsion_type])
    gamma_torsion = np.where(np.array(d_torsion) == 1, 0.0, jnp.pi)

    improper_idx = np.array([[int(l.split()[2]) - 1, int(l.split()[3]) - 1, int(l.split()[4]) - 1, int(l.split()[5]) - 1] for l in data['Impropers']])
    improper_type = np.array([int(l.split()[1]) - 1 for l in data['Impropers']])
    k_improper = np.array([improper_raw[t][0] for t in improper_type])
    d_improper = np.array([improper_raw[t][1] for t in improper_type])
    n_improper = np.array([improper_raw[t][2] for t in improper_type])
    gamma_improper = np.where(np.array(d_improper) == 1, 0.0, jnp.pi)

    def get_nonbonded_pairs_with_14_mask(num_atoms, bond_idx, angle_idx, torsion_idx, improper_idx):
        i_vals, j_vals = np.triu_indices(num_atoms, k=1)
        all_pairs = np.stack([i_vals, j_vals], axis=1)

        bond_pairs = np.sort(bond_idx[:, [0, 1]], axis=1)
        angle_pairs = np.sort(angle_idx[:, [0, 2]], axis=1)
        excluded = np.unique(np.vstack([bond_pairs, angle_pairs]), axis=0)

        all_set = set(map(tuple, all_pairs.tolist()))
        excluded_set = set(map(tuple, excluded.tolist()))
        nb_pairs = np.array(list(all_set - excluded_set))
        nb_pairs = nb_pairs[np.lexsort((nb_pairs[:,1], nb_pairs[:,0]))]

        one_four = np.sort(torsion_idx[:, [0, 3]], axis=1)
        one_four_set = set(map(tuple, one_four.tolist()))
        is_14 = np.array([tuple(p) in one_four_set for p in nb_pairs], dtype=bool)

        return jnp.array(nb_pairs), jnp.array(is_14)

    pair_indices, is_14_mask = get_nonbonded_pairs_with_14_mask(num_atoms, bond_idx, angle_idx, torsion_idx, improper_idx)
    box = jnp.array([44.0, 44.0, 44.0])  # Ideally parse from file header

    return (
        jnp.array(positions), 
        (jnp.array(bond_idx), jnp.array(k_b), jnp.array(r0)),
        (jnp.array(angle_idx), jnp.array(k_theta), jnp.array(theta0)),
        (jnp.array(torsion_idx), jnp.array(k_torsion), jnp.array(d_torsion), jnp.array(n_torsion), jnp.array(gamma_torsion)),
        (jnp.array(improper_idx), jnp.array(k_improper), jnp.array(d_improper), jnp.array(n_improper), jnp.array(gamma_improper)),
        (jnp.array(charges), jnp.array(sigmas), jnp.array(epsilons), jnp.array(pair_indices), is_14_mask),
        jnp.array(molecule_id), box,
        jnp.array(masses)  # 🔹 New return value
    )

