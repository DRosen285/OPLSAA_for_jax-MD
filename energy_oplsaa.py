import jax
import jax.numpy as jnp
from jax import vmap
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
    neighbor_fn = partition.neighbor_list(displacement_fn, box, r_cut, dr_threshold=dr_threshold,mask=True)

    def make_is_14_lookup(pair_indices, molecule_id, is_14_mask, num_atoms):
        # Determine if both atoms in the pair belong to the same molecule
        same_mol = molecule_id[pair_indices[:, 0]] == molecule_id[pair_indices[:, 1]]
        valid_mask = is_14_mask & same_mol

        # Initialize the table
        is_14_table = jnp.zeros((num_atoms, num_atoms), dtype=bool)

        # Define a function to update the table for a single valid pair
        def update_table(table, pair_and_mask):
            pair, mask = pair_and_mask
            def set_pair(t):
                t = t.at[pair[0], pair[1]].set(True)
                t = t.at[pair[1], pair[0]].set(True)
                return t
            return jax.lax.cond(mask, set_pair, lambda t: t, table)

        # Use scan to apply updates across all pairs
        is_14_table = jax.lax.fori_loop(
           0, pair_indices.shape[0],
           lambda i, t: update_table(t, (pair_indices[i], valid_mask[i])),
           is_14_table
           )

        return is_14_table

    def soft_lj(r, sigma, epsilon, cap=1000.0):
        r_safe = jnp.where(r < 1e-6, 1e-6, r)
        lj = 4 * epsilon * ((sigma / r_safe)**12 - (sigma / r_safe)**6)
        return  cap * jnp.tanh(lj / cap)

    def dihedral(p0, p1, p2, p3):
        b0 = displacement_fn(p1, p0)
        b1 = displacement_fn(p2, p1)
        b2 = displacement_fn(p3, p2)
        n1 = jnp.cross(b0, b1)
        n2 = jnp.cross(b1, b2)
        n1 /= jnp.linalg.norm(n1)
        n2 /= jnp.linalg.norm(n2)
        cos_phi = jnp.dot(n1, n2)
        phi = jnp.arccos(jnp.clip(cos_phi, -1.0, 1.0))
        return phi

    def dihedral_signed(p0, p1, p2, p3):
        b0 = displacement_fn(p1, p0)
        b1 = displacement_fn(p2, p1)
        b2 = displacement_fn(p3, p2)
        b1 /= jnp.linalg.norm(b1, axis=-1, keepdims=True)
        v = b0 - jnp.sum(b0 * b1, axis=-1, keepdims=True) * b1
        w = b2 - jnp.sum(b2 * b1, axis=-1, keepdims=True) * b1
        x = jnp.sum(v * w, axis=-1)
        y = jnp.sum(jnp.cross(b1, v) * w, axis=-1)
        return jnp.arctan2(y, x)

    @jax.jit
    def bonded_and_lj_energy(positions, nlist):
        num_atoms = positions.shape[0]
        is_14_table = make_is_14_lookup(pair_indices, molecule_id ,is_14_mask, num_atoms)

        exclusion_mask = jnp.zeros((num_atoms, num_atoms), dtype=bool)
        exclusion_mask = exclusion_mask.at[bond_idx[:, 0], bond_idx[:, 1]].set(True)
        exclusion_mask = exclusion_mask.at[bond_idx[:, 1], bond_idx[:, 0]].set(True)
        exclusion_mask = exclusion_mask.at[angle_idx[:, 0], angle_idx[:, 2]].set(True)
        exclusion_mask = exclusion_mask.at[angle_idx[:, 2], angle_idx[:, 0]].set(True)

        # Bonded terms
        disp_bond = vmap(displacement_fn)(positions[bond_idx[:, 0]], positions[bond_idx[:, 1]])
        r_bond = jnp.linalg.norm(disp_bond, axis=-1)
        E_bond = jnp.sum(k_b * (r_bond - r0) ** 2)

        i, j, k = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
        rij = vmap(displacement_fn)(positions[i], positions[j])
        rkj = vmap(displacement_fn)(positions[k], positions[j])
        norm_rij = jnp.linalg.norm(rij, axis=-1)
        norm_rkj = jnp.linalg.norm(rkj, axis=-1)
        cos_theta = jnp.sum(rij * rkj, axis=-1) / (norm_rij * norm_rkj)
        theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
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
            same_atom = (i == j)
            excluded = exclusion_mask[i, j]
            disp = displacement_fn(positions[i], positions[j])
            r = jnp.linalg.norm(disp)
            r_safe = jnp.where(r < 1e-6, 1e-6, r)

       
            sigma_ij = jnp.sqrt(sigmas[i] * sigmas[j])
            epsilon_ij = jnp.sqrt(epsilons[i] * epsilons[j])

            if use_soft_lj:
                lj = soft_lj(r_safe, sigma_ij, epsilon_ij, cap=lj_cap)
            else:
                lj = 4 * epsilon_ij * ((sigma_ij / r_safe) ** 12 - (sigma_ij / r_safe) ** 6)
             
            if use_shift_lj:
               lj_shift = 4 * epsilon_ij * ((sigma_ij / r_cut) ** 12 - (sigma_ij / r_cut) ** 6) 
               scale_lj = jnp.where(is_14_table[i, j], 0.5, 1.0)
               include = (~same_atom) & (~excluded) & (r_safe < r_cut)
               return jnp.where(include, scale_lj * (lj-lj_shift), 0.0)
            else:
               scale_lj = jnp.where(is_14_table[i, j], 0.5, 1.0)
               include = (~same_atom) & (~excluded) & (r_safe < r_cut)
               return jnp.where(include, scale_lj * lj, 0.0)


        def sum_over_neighbors(i):
            neighbors = nlist.idx[i]                          # shape (N_max,)
            mask = neighbors < num_atoms                           # valid neighbor mask
            safe_neighbors = jnp.where(mask, neighbors, 0)    # dummy value is 0

            def compute_if_valid(j, valid):
                return jnp.where(valid, compute_pair_energy(i, j), 0.0)

            pair_energies = vmap(compute_if_valid)(safe_neighbors, mask)
            
            return jnp.sum(pair_energies)
            
        E_nb = 0.5 * jnp.sum(vmap(sum_over_neighbors)(jnp.arange(num_atoms)))
        return  E_bond + E_angle + E_torsion + E_improper + E_nb

    return bonded_and_lj_energy, neighbor_fn, displacement_fn
