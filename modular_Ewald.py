import jax
import jax.numpy as jnp
import jax.scipy
from jax import vmap
from jax_md import space
from jax.numpy.fft import fftn, fftfreq
import itertools

class CoulombHandler:
    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, box):
        raise NotImplementedError


class CutoffCoulomb(CoulombHandler):
    def __init__(self, r_cut, use_erfc=False, alpha=0.3):
        self.r_cut = r_cut
        self.use_erfc = use_erfc
        self.alpha = alpha
        self.prefactor = 332.06371

    def pair_energy(self, qi, qj, r):
        if self.use_erfc:
            return self.prefactor * (qi * qj / r) * jax.scipy.special.erfc(self.alpha * r)
        else:
            return self.prefactor * (qi * qj / r)

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

        pairwise_energy = vmap(lambda i: jnp.sum(vmap(lambda j: compute_pair_energy(i, j))(jnp.arange(num_atoms))))(jnp.arange(num_atoms))
        return 0.5 * jnp.sum(pairwise_energy)



class EwaldCoulomb(CoulombHandler):
    def __init__(self, alpha=0.23, kmax=5, r_cut=15.0):
        self.alpha = alpha
        self.kmax = kmax
        self.r_cut = r_cut
        self.prefactor = 332.06371

    def lammps_erfc(self, x):
        p = 0.3275911
        A1 = 0.254829592
        A2 = -0.284496736
        A3 = 1.421413741
        A4 = -1.453152027
        A5 = 1.061405429
        
        t = 1.0 / (1.0 + p * x)
        expm2 = jnp.exp(-x * x)
        poly = (((A5 * t + A4) * t + A3) * t + A2) * t + A1
        return t * poly * expm2

    def reciprocal_energy(self, positions, charges, box):
        vol = jnp.prod(box)
        kx = jnp.arange(-self.kmax, self.kmax + 1)
        ky = jnp.arange(-self.kmax, self.kmax + 1)
        kz = jnp.arange(-self.kmax, self.kmax + 1)
        KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
        kvecs = jnp.stack([KX, KY , KZ], axis=-1).reshape(-1, 3)

        def compute_term(k):
            is_zero = jnp.all(k == 0)
            k_cart = 2 * jnp.pi * k / box
            k2 = jnp.dot(k_cart, k_cart)
            rho_k = jnp.sum(charges * jnp.exp(1j * jnp.dot(positions, k_cart)))
            factor = jnp.where(is_zero, 0.0, jnp.exp(-k2 / (4 * self.alpha ** 2)) / k2)
            return (4 * jnp.pi / vol) * factor * jnp.abs(rho_k) ** 2

        energy_terms = vmap(compute_term)(kvecs)
        return 0.5 * jnp.sum(energy_terms)

    def self_energy(self, charges):
        return -self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges ** 2) * self.prefactor

    def real_energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, nlist):
        num_atoms = positions.shape[0]
        max_neighbors = nlist.idx.shape[1]

        # Shape (N, M)
        idx_i = jnp.repeat(jnp.arange(num_atoms)[:, None], max_neighbors, axis=1)
        idx_j = nlist.idx  # (N, M)

    # Valid neighbor entries
        valid = idx_j >= 0
        ri = positions[idx_i]  # (N, M, 3)
        rj = positions[idx_j]  # (N, M, 3)

        qi = charges[idx_i]  # (N, M)
        qj = charges[idx_j]  # (N, M)

        batched_displacement = jax.vmap(jax.vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
        disp = batched_displacement(ri, rj)
        r = jnp.linalg.norm(disp, axis=-1)  # (N, M)

        same = idx_i == idx_j  # (N, M)
        excluded = exclusion_mask[idx_i, idx_j]
        is_14 = is_14_table[idx_i, idx_j]
        r_lt_cut = r < self.r_cut

    # Include only valid, non-self, within cutoff
        include = valid & (~same) & r_lt_cut

    # Coulomb scaling: 0.5 for 1-4, 1.0 otherwise, 0.0 if excluded
        factor_coul = jnp.where(is_14, 0.5, 1.0)
        factor_coul = jnp.where(excluded, 0.0, factor_coul)

    # Erfc term
        erfc_val = self.lammps_erfc(self.alpha * r)

    # Energy expression
        energy_raw = self.prefactor * qi * qj / r * (erfc_val - (1.0 - factor_coul))
        energy = jnp.where(include, energy_raw, 0.0)

        total_energy = jnp.sum(energy) * 0.5
        return total_energy

    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, box,nlist):
        e_real = self.real_energy(positions, charges, displacement_fn, exclusion_mask, is_14_table,nlist)
        e_recip = self.reciprocal_energy(positions, charges, box)
        e_self = self.self_energy(charges)
        e_total = e_real + e_recip + e_self
        return e_real, e_recip, e_self, e_total


class PME_Coulomb:
    def __init__(self, grid_size=32, alpha=0.3, r_cut=15.0):
        self.grid_size = grid_size
        self.alpha = alpha
        self.r_cut = r_cut
        self.prefactor = 332.06371  # kcal/mol units

    def structure_factor(self, charges, positions, box):
        scaled_pos = positions / box * self.grid_size
        base = jnp.floor(scaled_pos).astype(int)
        frac = scaled_pos - base

        def deposit_single(charge, base, frac):
            ix = (base[0] + jnp.array([0, 1])) % self.grid_size
            iy = (base[1] + jnp.array([0, 1])) % self.grid_size
            iz = (base[2] + jnp.array([0, 1])) % self.grid_size

            wx = jnp.array([1 - frac[0], frac[0]])
            wy = jnp.array([1 - frac[1], frac[1]])
            wz = jnp.array([1 - frac[2], frac[2]])

            grid = jnp.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=jnp.complex64)

            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        weight = charge * wx[dx] * wy[dy] * wz[dz]
                        grid = grid.at[ix[dx], iy[dy], iz[dz]].add(weight)
            return grid

        grids = vmap(deposit_single)(charges, base, frac)
        return jnp.sum(grids, axis=0)

    def reciprocal_energy(self, rho_k, box):
        vol = jnp.prod(box)
        gx = fftfreq(self.grid_size) * self.grid_size
        gy = fftfreq(self.grid_size) * self.grid_size
        gz = fftfreq(self.grid_size) * self.grid_size
        Gx, Gy, Gz = jnp.meshgrid(gx, gy, gz, indexing='ij')
        G2 = (2 * jnp.pi) ** 2 * (Gx**2 / box[0]**2 + Gy**2 / box[1]**2 + Gz**2 / box[2]**2)

        mask = G2 > 0
        factor = jnp.where(mask, jnp.exp(-G2 / (4 * self.alpha ** 2)) / G2, 0.0)
        rho_sq = jnp.abs(rho_k) ** 2
        energy = (4 * jnp.pi / vol) * jnp.sum(factor * rho_sq)
        return 0.5 * self.prefactor * energy

    def self_energy(self, charges):
        return -self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges ** 2) * self.prefactor

    def real_energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, nlist):
        num_atoms = positions.shape[0]
        max_neighbors = nlist.idx.shape[1]

        # Shape (N, M)
        idx_i = jnp.repeat(jnp.arange(num_atoms)[:, None], max_neighbors, axis=1)
        idx_j = nlist.idx  # (N, M)

    # Valid neighbor entries
        valid = idx_j >= 0
        ri = positions[idx_i]  # (N, M, 3)
        rj = positions[idx_j]  # (N, M, 3)

        qi = charges[idx_i]  # (N, M)
        qj = charges[idx_j]  # (N, M)

        batched_displacement = jax.vmap(jax.vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
        disp = batched_displacement(ri, rj)
        r = jnp.linalg.norm(disp, axis=-1)  # (N, M)

        same = idx_i == idx_j  # (N, M)
        excluded = exclusion_mask[idx_i, idx_j]
        is_14 = is_14_table[idx_i, idx_j]
        r_lt_cut = r < self.r_cut

    # Include only valid, non-self, within cutoff
        include = valid & (~same) & r_lt_cut

    # Coulomb scaling: 0.5 for 1-4, 1.0 otherwise, 0.0 if excluded
        factor_coul = jnp.where(is_14, 0.5, 1.0)
        factor_coul = jnp.where(excluded, 0.0, factor_coul)

    # Erfc term
        erfc_val =  jax.scipy.special.erfc(self.alpha * r)  #self.lammps_erfc(self.alpha * r)

    # Energy expression
        energy_raw = self.prefactor * qi * qj / r * (erfc_val - (1.0 - factor_coul))
        energy = jnp.where(include, energy_raw, 0.0)

        total_energy = jnp.sum(energy) * 0.5
        return total_energy


    def energy(self, positions, charges, displacement_fn, exclusion_mask, is_14_table, box,nlist):
        rho_real = self.structure_factor(charges, positions, box)
        rho_k = fftn(rho_real)
        E_recip = self.reciprocal_energy(rho_k, box)
        E_self = self.self_energy(charges)
        E_real = self.real_energy(positions, charges, displacement_fn, exclusion_mask, is_14_table,nlist)
        return E_real , E_recip, E_self, E_real  + E_recip + E_self



def make_is_14_lookup(pair_indices, is_14_mask, num_atoms):
    is_14_table = jnp.zeros((num_atoms, num_atoms), dtype=bool)
    is_14_table = is_14_table.at[pair_indices[:, 0], pair_indices[:, 1]].set(is_14_mask)
    is_14_table = is_14_table.at[pair_indices[:, 1], pair_indices[:, 0]].set(is_14_mask)
    return is_14_table

