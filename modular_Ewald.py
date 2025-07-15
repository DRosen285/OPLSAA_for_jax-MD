import jax
import jax.numpy as jnp
import jax.scipy
from jax import vmap
from jax_md import space, partition
from jax.numpy.fft import fftn, ifftn, fftfreq

class CoulombHandler:
    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, box):
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

    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, box):
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
        kmax = self.kmax
        kx = jnp.arange(-kmax, kmax+1)
        ky = jnp.arange(-kmax, kmax+1)
        kz = jnp.arange(-kmax, kmax+1)
        KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
        kvecs = jnp.stack([KX, KY, KZ], axis=-1)
        kvecs = kvecs.reshape(-1, 3)
        kvecs = kvecs[jnp.any(kvecs != 0, axis=1)]  # remove (0,0,0)

        energy_terms = []
        for k in kvecs:
            k_cart = 2 * jnp.pi * k / box
            k2 = jnp.dot(k_cart, k_cart)
            rho_k = jnp.sum(charges * jnp.exp(1j * jnp.dot(positions, k_cart)))
            factor = jnp.exp(-k2 / (4 * self.alpha**2)) / k2
            term = (4 * jnp.pi / vol) * factor * jnp.abs(rho_k)**2
            energy_terms.append(term)
        return 0.5 * jnp.sum(jnp.array(energy_terms))

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

class PME_Coulomb(CoulombHandler):
    def __init__(self, grid_size=32, alpha=0.3):
        self.grid_size = grid_size
        self.alpha = alpha

    def structure_factor(self, charges, positions, box):
        grid = jnp.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=jnp.complex64)
        scaled_pos = positions / box * self.grid_size

        def deposit_b_spline(charge, pos):
            base = jnp.floor(pos).astype(int)
            frac = pos - base
            result = grid
            for dx in [0, 1]:
                wx = 1 - frac[0] if dx == 0 else frac[0]
                ix = (base[0] + dx) % self.grid_size
                for dy in [0, 1]:
                    wy = 1 - frac[1] if dy == 0 else frac[1]
                    iy = (base[1] + dy) % self.grid_size
                    for dz in [0, 1]:
                        wz = 1 - frac[2] if dz == 0 else frac[2]
                        iz = (base[2] + dz) % self.grid_size
                        weight = charge * wx * wy * wz
                        result = result.at[ix, iy, iz].add(weight)
            return result

        for i in range(charges.shape[0]):
            grid = deposit_b_spline(charges[i], scaled_pos[i])

        return grid

    def reciprocal_energy(self, rho_k, box):
        vol = jnp.prod(box)
        gx = fftfreq(self.grid_size) * self.grid_size
        gy = fftfreq(self.grid_size) * self.grid_size
        gz = fftfreq(self.grid_size) * self.grid_size
        Gx, Gy, Gz = jnp.meshgrid(gx, gy, gz, indexing='ij')
        G2 = (2 * jnp.pi)**2 * (Gx**2 / box[0]**2 + Gy**2 / box[1]**2 + Gz**2 / box[2]**2)

        mask = G2 > 0
        factor = jnp.where(mask, jnp.exp(-G2 / (4 * self.alpha**2)) / G2, 0.0)
        rho_sq = jnp.abs(rho_k) ** 2
        energy = (4 * jnp.pi / vol) * jnp.sum(factor * rho_sq)
        return 0.5 * 332.06371 * energy

    def self_energy(self, charges):
        return -self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges ** 2) * 332.06371

    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, box):
        rho_real = self.structure_factor(charges, positions, box)
        rho_k = fftn(rho_real)
        E_recip = self.reciprocal_energy(rho_k, box)
        E_self = self.self_energy(charges)
        return E_recip + E_self

def make_is_14_lookup(pair_indices, is_14_mask, num_atoms):
    is_14_table = jnp.zeros((num_atoms, num_atoms), dtype=bool)
    is_14_table = is_14_table.at[pair_indices[:, 0], pair_indices[:, 1]].set(is_14_mask)
    is_14_table = is_14_table.at[pair_indices[:, 1], pair_indices[:, 0]].set(is_14_mask)
    return is_14_table

