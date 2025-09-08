import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax_md import space, partition
import numpy as np
from jax import debug

def optimized_opls_aa_energy_with_nlist_modular(
    bonds, angles, torsions, impropers,
    nonbonded, molecule_id, box, r_cut=15.0, dr_threshold=0.5,
    use_soft_lj=False, lj_cap=1000.0, use_shift_lj=False
):
    box = np.asarray(box)

    bond_idx, k_b, r0 = bonds
    angle_idx, k_theta, theta0 = angles
    torsion_idx, k_torsion, _, n_torsion, gamma_torsion = torsions
    improper_idx, k_improper, _, n_improper, gamma_improper = impropers
    charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded

    displacement_fn, shift_fn = space.periodic(box)
    neighbor_fn = partition.neighbor_list(displacement_fn, box, r_cut, dr_threshold=dr_threshold, mask=True)

    def safe_norm(x, axis=-1, epsilon=1e-6, keepdims=False):
        squared = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
        norm = jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims))
        return jnp.maximum(norm, epsilon)
 
    def safe_distance(x, y, epsilon=1e-4):
        disp = displacement_fn(x, y)
        return safe_norm(disp, epsilon=epsilon)

    def safe_arccos(x, epsilon=1e-6):
        x = jnp.clip(x, -1.0 + epsilon, 1.0 - epsilon)
        return jnp.arccos(x)

    def normalize(x, axis=-1, epsilon=1e-6):
        norm = safe_norm(x, axis=axis, epsilon=epsilon, keepdims=True)
        return x / norm


    def make_is_14_lookup(pair_indices, molecule_id, is_14_mask, num_atoms):
        same_mol = molecule_id[pair_indices[:, 0]] == molecule_id[pair_indices[:, 1]]
        valid_mask = is_14_mask & same_mol
        is_14_table = jnp.zeros((num_atoms, num_atoms), dtype=bool)

        def update_table(table, pair_and_mask):
            pair, mask = pair_and_mask
            def set_pair(t):
                t = t.at[pair[0], pair[1]].set(True)
                t = t.at[pair[1], pair[0]].set(True)
                return t
            return jax.lax.cond(mask, set_pair, lambda t: t, table)

        is_14_table = jax.lax.fori_loop(
            0, pair_indices.shape[0],
            lambda i, t: update_table(t, (pair_indices[i], valid_mask[i])),
            is_14_table
        )
        return is_14_table

    def soft_lj(r, sigma, epsilon, cap=1000.0):
        lj = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        return cap * jnp.tanh(lj / cap)

    def dihedral(p0, p1, p2, p3):
        b0 = displacement_fn(p1, p0)
        b1 = displacement_fn(p2, p1)
        b2 = displacement_fn(p3, p2)

        n1 = jnp.cross(b0, b1)
        n2 = jnp.cross(b1, b2)

        n1 = normalize(n1)
        n2 = normalize(n2)

        cos_phi = jnp.sum(n1 * n2, axis=-1)
        phi = safe_arccos(cos_phi)
        return phi


    def dihedral_signed(p0, p1, p2, p3):
        b0 = displacement_fn(p1, p0)
        b1 = displacement_fn(p2, p1)
        b2 = displacement_fn(p3, p2)

        b1 = normalize(b1)

        v = b0 - jnp.sum(b0 * b1, axis=-1, keepdims=True) * b1
        w = b2 - jnp.sum(b2 * b1, axis=-1, keepdims=True) * b1

        x = jnp.sum(v * w, axis=-1)
        y = jnp.sum(jnp.cross(b1, v) * w, axis=-1)
        return jnp.arctan2(y, x)


    def angle_between(v1, v2, epsilon=1e-6):
        v1_norm = normalize(v1, epsilon=epsilon)
        v2_norm = normalize(v2, epsilon=epsilon)
        dot = jnp.sum(v1_norm * v2_norm, axis=-1)
        return safe_arccos(dot, epsilon=epsilon)

    def bonded_and_lj_energy(positions, nlist):
        num_atoms = positions.shape[0]
        is_14_table = make_is_14_lookup(pair_indices, molecule_id, is_14_mask, num_atoms)

        exclusion_mask = jnp.zeros((num_atoms, num_atoms), dtype=bool)
        exclusion_mask = exclusion_mask.at[bond_idx[:, 0], bond_idx[:, 1]].set(True)
        exclusion_mask = exclusion_mask.at[bond_idx[:, 1], bond_idx[:, 0]].set(True)
        exclusion_mask = exclusion_mask.at[angle_idx[:, 0], angle_idx[:, 2]].set(True)
        exclusion_mask = exclusion_mask.at[angle_idx[:, 2], angle_idx[:, 0]].set(True)

        # Bonded terms
        r_bond=vmap(safe_distance)(positions[bond_idx[:, 0]], positions[bond_idx[:, 1]])
        E_bond = jnp.sum(k_b * (r_bond - r0) ** 2)

        i, j, k = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
        rij = vmap(displacement_fn)(positions[i], positions[j])
        rkj = vmap(displacement_fn)(positions[k], positions[j])

        theta = angle_between(rij, rkj)
        E_angle = jnp.sum(k_theta * (theta - theta0) ** 2)

        phi = vmap(dihedral)(
            positions[torsion_idx[:, 0]],
            positions[torsion_idx[:, 1]],
            positions[torsion_idx[:, 2]],
            positions[torsion_idx[:, 3]]
        )
        E_torsion = jnp.sum(k_torsion * (1 + jnp.cos(n_torsion * phi - gamma_torsion)))

        psi = vmap(dihedral_signed)(
            positions[improper_idx[:, 0]],
            positions[improper_idx[:, 1]],
            positions[improper_idx[:, 2]],
            positions[improper_idx[:, 3]]
        )
        E_improper = jnp.sum(k_improper * (1 + jnp.cos(n_improper * psi - gamma_improper)))

        # Non-bonded pairwise LJ energy
        
        def compute_pair_energy(i, j):
            j_clamped = jnp.minimum(j, num_atoms - 1)
            same_atom = (i == j_clamped)
            excluded = exclusion_mask[i, j_clamped]

            # Compute displacement and distance
            disp = displacement_fn(positions[i], positions[j_clamped])
            r2 = jnp.sum(disp ** 2)
            r2_safe = jnp.maximum(r2, 1e-4)  # avoid divide-by-zero or NaN in sqrt
            r = jnp.sqrt(r2_safe)

            # Sigma and epsilon
            sigma_ij = jnp.sqrt(sigmas[i] * sigmas[j_clamped])
            epsilon_ij = jnp.sqrt(epsilons[i] * epsilons[j_clamped])

            # LJ calculation
            def lj_energy(r2, r, sigma_ij, epsilon_ij):
                sr = sigma_ij / jnp.sqrt(r2)
                sr6 = sr ** 6
                lj_val = 4.0 * epsilon_ij * (sr6 ** 2 - sr6)

                if use_soft_lj:
                # Cap energy smoothly
                  return lj_cap * jnp.tanh(lj_val / lj_cap)

                elif use_shift_lj:
                # Shift so energy goes to zero at cutoff
                     sr_cut = sigma_ij / r_cut
                     sr6_cut = sr_cut ** 6
                     lj_cut = 4.0 * epsilon_ij * (sr6_cut ** 2 - sr6_cut)
                     return lj_val - lj_cut

                else:
                    return lj_val

            # Apply scaling
            scale_lj = jnp.where(is_14_table[i, j_clamped], 0.5, 1.0)

            # Final include mask
            include = (~same_atom) & (~excluded) & (r < r_cut)

            # Apply masking safely
            lj = lj_energy(r2_safe, r, sigma_ij, epsilon_ij)
            return jnp.where(include, scale_lj * lj, 0.0)

        def sum_over_neighbors(i):
            neighbors = nlist.idx[i]
            mask = neighbors < num_atoms

            def energy_fn(j, valid):
                return jnp.where(valid, compute_pair_energy(i, j), 0.0)

            pair_energies = jax.vmap(energy_fn)(neighbors, mask)
            return jnp.sum(pair_energies)


        E_nb = 0.5 * jnp.sum(jax.vmap(sum_over_neighbors)(jnp.arange(num_atoms)))


        return E_bond, E_angle, E_torsion, E_improper, E_nb, E_bond + E_angle + E_torsion + E_improper + E_nb 

    return bonded_and_lj_energy, neighbor_fn, displacement_fn

