import jax
import jax.numpy as jnp
import jax.scipy
from jax import vmap
from jax_md import space, partition

class CoulombHandler:
    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table):
        raise NotImplementedError

class CutoffCoulomb(CoulombHandler):
    def __init__(self, r_cut, use_erfc=False, alpha=0.3):
        self.r_cut = r_cut
        self.use_erfc = use_erfc
        self.alpha = alpha

    def pair_energy(self, qi, qj, r):
        prefactor = 332.06371
        if self.use_erfc:
            return prefactor * (qi * qj / r) * jax.scipy.special.erfc(self.alpha * r)
        else:
            return prefactor * (qi * qj / r)

    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table):
        num_atoms = positions.shape[0]

        def compute_pair_energy(i, j):
            same = (i == j)
            excluded = exclusion_mask[i, j]
            disp = displacement_fn(positions[i], positions[j])
            r = jnp.linalg.norm(disp)
            include = (~same) & (~excluded) & (r < self.r_cut)

            qi, qj = charges[i], charges[j]
            is_14 = is_14_table[i, j]
            scale = jnp.where(is_14, 0.5, 1.0)
            energy = scale * self.pair_energy(qi, qj, r)
            return jnp.where(include, energy, 0.0)

        E = 0.0
        for i in range(num_atoms):
            E += jnp.sum(vmap(lambda j: compute_pair_energy(i, j))(jnp.arange(num_atoms)))
        return 0.5 * E

class EwaldCoulomb(CoulombHandler):
    def __init__(self, alpha=0.3, kmax=5, r_cut=15.0):
        self.alpha = alpha
        self.kmax = kmax
        self.r_cut = r_cut

    def reciprocal_energy(self, positions, charges, box):
        vol = jnp.prod(box)
        k_vectors = []
        for nx in range(-self.kmax, self.kmax+1):
            for ny in range(-self.kmax, self.kmax+1):
                for nz in range(-self.kmax, self.kmax+1):
                    if nx == ny == nz == 0:
                        continue
                    k = 2 * jnp.pi * jnp.array([nx, ny, nz]) / box
                    k2 = jnp.dot(k, k)
                    if k2 == 0:
                        continue
                    rho_k = jnp.sum(charges * jnp.exp(1j * jnp.dot(positions, k)))
                    factor = jnp.exp(-k2 / (4 * self.alpha**2)) / k2
                    k_vectors.append((4 * jnp.pi / vol) * factor * jnp.abs(rho_k)**2)
        return 0.5 * jnp.sum(jnp.stack(k_vectors))

    def self_energy(self, charges):
        return -self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges ** 2) * 332.06371

    def real_energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table):
        num_atoms = positions.shape[0]

        def compute_pair_energy(i, j):
            same = (i == j)
            excluded = exclusion_mask[i, j]
            disp = displacement_fn(positions[i], positions[j])
            r = jnp.linalg.norm(disp)
            include = (~same) & (~excluded) & (r < self.r_cut)
            prefactor = 332.06371
            energy = prefactor * charges[i] * charges[j] / r * jax.scipy.special.erfc(self.alpha * r)
            is_14 = is_14_table[i, j]
            scale = jnp.where(is_14, 0.5, 1.0)
            return jnp.where(include, scale * energy, 0.0)

        E = 0.0
        for i in range(num_atoms):
            E += jnp.sum(vmap(lambda j: compute_pair_energy(i, j))(jnp.arange(num_atoms)))
        return 0.5 * E

    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, box):
        return (self.real_energy(positions, charges, displacement_fn, exclusion_mask, is_14_table)
              + self.reciprocal_energy(positions, charges, box)
              + self.self_energy(charges))

def make_is_14_lookup(pair_indices, is_14_mask, num_atoms):
    is_14_table = jnp.zeros((num_atoms, num_atoms), dtype=bool)
    is_14_table = is_14_table.at[pair_indices[:, 0], pair_indices[:, 1]].set(is_14_mask)
    is_14_table = is_14_table.at[pair_indices[:, 1], pair_indices[:, 0]].set(is_14_mask)
    return is_14_table

