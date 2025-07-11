import jax.numpy as jnp
from jax import vmap
from jax_md import space
import numpy as np

def opls_aa_energy(
    bonds, angles, torsions, impropers,
    nonbonded,
    box=None
):
    """Returns a function energy_fn(positions) that computes the total OPLS-AA energy."""
    bond_idx, k_b, r0 = bonds
    angle_idx, k_theta, theta0 = angles
    torsion_idx, k_torsion, d_torsion, n_torsion, gamma_torsion = torsions
    improper_idx, k_improper, d_improper, n_improper, gamma_improper = impropers
    charges, sigmas, epsilons,pair_indices, is_14_mask = nonbonded

    def safe_norm(x, axis=-1, epsilon=1e-8):
        return jnp.sqrt(jnp.sum(x ** 2, axis=axis) + epsilon)

    def dihedral(p0, p1, p2, p3):
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        #b1 /= safe_norm(b1)
        #v = b0 - jnp.dot(b0, b1) * b1
        #w = b2 - jnp.dot(b2, b1) * b1
        #x = jnp.dot(v, w)
        #y = jnp.dot(jnp.cross(b1, v), w)
        n1 = jnp.cross(b0, b1)
        n2 = jnp.cross(b1, b2)
        n1 /= jnp.linalg.norm(n1)
        n2 /= jnp.linalg.norm(n2)
        cos_phi = jnp.dot(n1, n2)
        phi = jnp.arccos(jnp.clip(cos_phi, -1.0, 1.0)) # returns unasigned angle in radians(no difference between clockwise, counter clockwise)
        #return jnp.arctan2(y, x)
        return phi

    def dihedral_signed(p0, p1, p2, p3):
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= jnp.linalg.norm(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1, v), w)
        return jnp.arctan2(y, x)  # returns signed angle in radians


    def energy_fn(positions):
        # Bonds
        rij = positions[bond_idx[:, 0]] - positions[bond_idx[:, 1]]
        r = jnp.linalg.norm(rij, axis=-1)
        #E_bond = jnp.sum(0.5 * k_b * (r - r0) ** 2)
        E_bond = jnp.sum(k_b * (r - r0) ** 2)
        # Angles
        i, j, k = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
        rij = positions[i] - positions[j]
        rkj = positions[k] - positions[j]
        cos_theta = jnp.sum(rij * rkj, axis=-1) / (
            jnp.linalg.norm(rij, axis=-1) * jnp.linalg.norm(rkj, axis=-1)
        )
        theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
        #E_angle = jnp.sum(0.5 * k_theta * (theta - theta0) ** 2)
        E_angle = jnp.sum(k_theta * (theta - theta0) ** 2)

        # Torsions
        # torsion_idx: (num_torsions, 4)
        ti, tj, tk, tl = torsion_idx[:, 0], torsion_idx[:, 1], torsion_idx[:, 2], torsion_idx[:, 3]
        phi = vmap(dihedral)(positions[ti], positions[tj], positions[tk], positions[tl])  # shape: (num_torsions,)
        # Parameters per torsion (shape: (num_torsions,))
        # LAMMPS harmonic-style energy
        E_torsion = k_torsion * (1 +  jnp.cos(n_torsion * phi - gamma_torsion)) 
        E_torsion = jnp.sum(E_torsion)

        # Impropers
        ii, ij, ik, il = improper_idx[:, 0], improper_idx[:, 1], improper_idx[:, 2], improper_idx[:, 3]
        psi = vmap(dihedral_signed)(positions[ii], positions[ij], positions[ik], positions[il])
        #E_improper = jnp.sum(0.5 * k_psi * (psi - psi0) ** 2)
        E_improper = k_improper  * (1 +  jnp.cos(n_improper * psi - gamma_improper))
        E_improper = jnp.sum(E_improper)


        # Nonbonded
        pi, pj = pair_indices[:, 0], pair_indices[:, 1]
        rij = positions[pi] - positions[pj]
        if box is not None:
            rij = rij - box * jnp.round(rij / box)
        r = jnp.linalg.norm(rij, axis=-1)
        # Apply cutoff
        r_cut = 15.0
        mask = r < r_cut
        # Filter values based on the cutoff
        r = r[mask]
        pi = pi[mask]
        pj = pj[mask]
        is_14 = is_14_mask[mask]

        sigma_ij = jnp.sqrt(sigmas[pi] * sigmas[pj])
        epsilon_ij = jnp.sqrt(epsilons[pi] * epsilons[pj])
        lj = 4 * epsilon_ij * ((sigma_ij / r) ** 12 - (sigma_ij / r) ** 6)
        coulomb = 332.06371 * (charges[pi] * charges[pj]) / r

        # Apply 1–4 scaling
        lj_scale_14=0.5
        lj_scaled = jnp.where(is_14, lj * lj_scale_14, lj)
        coulomb_scale_14=0.5
        coulomb_scaled = jnp.where(is_14, coulomb * coulomb_scale_14, coulomb)

        E_nb = jnp.sum(lj_scaled + coulomb_scaled)
        return E_bond +E_angle + E_torsion + E_improper + E_nb

    return energy_fn

def opls_aa_neighbor_energy(
    bonds, angles, torsions, impropers,
    charges, sigmas, epsilons,
    box_size,
    neighbor_fn
):
    bond_idx, k_b, r0 = bonds
    angle_idx, k_theta, theta0 = angles
    torsion_idx, V, delta = torsions
    improper_idx, k_psi, psi0 = impropers

    disp_fn, _ = space.periodic(box_size)


def dihedral(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= jnp.linalg.norm(b1)
    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1
    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)
    return jnp.arctan2(y, x)


    def energy_fn(R, neighbor_list):
        # ----- Bonded Terms -----
        # Bonds
        rij = R[bond_idx[:, 0]] - R[bond_idx[:, 1]]
        r = jnp.linalg.norm(rij, axis=-1)
        E_bond = jnp.sum(0.5 * k_b * (r - r0) ** 2)

        # Angles
        i, j, k = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
        rij = R[i] - R[j]
        rkj = R[k] - R[j]
        cos_theta = jnp.sum(rij * rkj, axis=-1) / (
            jnp.linalg.norm(rij, axis=-1) * jnp.linalg.norm(rkj, axis=-1)
        )
        theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
        E_angle = jnp.sum(0.5 * k_theta * (theta - theta0) ** 2)

        # Torsions
        ti, tj, tk, tl = torsion_idx[:, 0], torsion_idx[:, 1], torsion_idx[:, 2], torsion_idx[:, 3]
        phi = vmap(dihedral)(R[ti], R[tj], R[tk], R[tl])
        delta_rad = jnp.deg2rad(delta)
        E_torsion = 0.0
        for n in range(1, 5):
            E_torsion += 0.5 * V[:, n - 1] * (1 + jnp.cos(n * phi - delta_rad[n - 1]))
        E_torsion = jnp.sum(E_torsion)

        # Improper
        ii, ij, ik, il = improper_idx[:, 0], improper_idx[:, 1], improper_idx[:, 2], improper_idx[:, 3]
        psi = vmap(dihedral)(R[ii], R[ij], R[ik], R[il])
        E_improper = jnp.sum(0.5 * k_psi * (psi - psi0) ** 2)

        # ----- Nonbonded Terms via neighbor list -----
        def pairwise_nb(i, j):
            dr = disp_fn(R[i], R[j])
            r = jnp.linalg.norm(dr)
            sig_ij = jnp.sqrt(sigmas[i] * sigmas[j])
            eps_ij = jnp.sqrt(epsilons[i] * epsilons[j])
            lj = 4 * eps_ij * ((sig_ij / r)**12 - (sig_ij / r)**6)
            coulomb = 138.935458 * charges[i] * charges[j] / r
            return lj + coulomb
        E_nb = jnp.sum(vmap(
            lambda i: jnp.sum(vmap(lambda j: pairwise_nb(i, j))(neighbor_fn.idx[i]))
        )(jnp.arange(R.shape[0])))

        # Total energy
        return E_bond + E_angle + E_torsion + E_improper + 0.5 * E_nb  # avoid double-counting

    return energy_fn

