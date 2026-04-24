"""Spin-1/2 Hamiltonians as sparse local-energy operators.

Each Hamiltonian exposes ``local_terms(s)`` which, given a batch of
configurations ``s ∈ {-1, +1}^{B × N}``, returns a tuple

    (s_conn,    # (B, M, N)   the M connected configurations of each row
     mels,      # (B, M)      the matrix elements ⟨s'|H|s⟩
     diag)      # (B,)        the diagonal contribution ⟨s|H|s⟩

so that ``E_loc(s) = diag + Σ_m mels[m] * ψ(s_conn[m]) / ψ(s)``.

Convention: spin variables are ``±1`` (eigenvalues of σᶻ).
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class TFIM:
    """Transverse-field Ising model.

        H = -J Σ_<ij> σᶻ_i σᶻ_j  -  h Σ_i σˣ_i

    The critical point in 1D is at ``h/J = 1``.
    """

    bonds: np.ndarray  # (E, 2)
    n_sites: int
    J: float = 1.0
    h: float = 1.0

    def local_terms(self, s: jnp.ndarray):
        bonds = jnp.asarray(self.bonds)
        si = s[..., bonds[:, 0]]
        sj = s[..., bonds[:, 1]]
        diag = -self.J * jnp.sum(si * sj, axis=-1)

        # Off-diagonal: each σˣ_i flips spin i and contributes -h.
        flips = jnp.eye(self.n_sites, dtype=s.dtype) * (-2) + 1  # (N, N) with -1 on diag
        # s_conn[b, i, k] = s[b, k] * flips[i, k]
        s_conn = s[..., None, :] * flips
        mels = -self.h * jnp.ones(s.shape[:-1] + (self.n_sites,), dtype=s.dtype)
        return s_conn, mels, diag


@dataclass
class HeisenbergJ1J2:
    """Antiferromagnetic Heisenberg model with optional J₂ frustration.

        H = J₁ Σ_<ij> S_i · S_j  +  J₂ Σ_<<ij>> S_i · S_j

    where ``S = σ/2``.  Off-diagonal terms swap antiparallel spins:

        H |...↑↓...⟩ = ½ J |...↓↑...⟩ + ¼ J σᶻ_i σᶻ_j |...⟩

    If ``marshall_sign`` is True we apply the Marshall rotation
    ``ψ(s) → (-1)^{N_↑ on B} ψ(s)`` which makes the unfrustrated ground
    state strictly positive — typically much easier to learn.
    """

    bonds_nn: np.ndarray
    bonds_nnn: np.ndarray
    n_sites: int
    sublattice: np.ndarray
    J1: float = 1.0
    J2: float = 0.0
    marshall_sign: bool = True

    def _diag_off(self, s: jnp.ndarray, bonds: jnp.ndarray, J: float):
        si = s[..., bonds[:, 0]]
        sj = s[..., bonds[:, 1]]
        diag = J * 0.25 * jnp.sum(si * sj, axis=-1)

        # Off-diagonal flip: only when σᶻ_i σᶻ_j = -1.
        # New state s' = s with bits i,j swapped == s with both flipped (since opposite).
        E = bonds.shape[0]
        N = self.n_sites
        # build an (E, N) mask that flips both bond endpoints
        idx = jnp.arange(N)
        flip_i = (idx[None, :] == bonds[:, 0:1]).astype(s.dtype)
        flip_j = (idx[None, :] == bonds[:, 1:2]).astype(s.dtype)
        flip = 1.0 - 2.0 * (flip_i + flip_j)  # (E, N): -1 at endpoints
        s_conn = s[..., None, :] * flip  # (..., E, N)
        active = (si * sj < 0).astype(s.dtype)  # antiparallel pairs only
        mels = J * 0.5 * active  # (..., E)

        if self.marshall_sign:
            # Marshall rotation ⇒ exchange picks up a (-1) on the active bonds.
            mels = -mels
        return s_conn, mels, diag

    def local_terms(self, s: jnp.ndarray):
        bnn = jnp.asarray(self.bonds_nn)
        bnnn = jnp.asarray(self.bonds_nnn)

        s1, m1, d1 = self._diag_off(s, bnn, self.J1)
        terms = [(s1, m1, d1)]
        if self.J2 != 0.0 and self.bonds_nnn.size > 0:
            s2, m2, d2 = self._diag_off(s, bnnn, self.J2)
            terms.append((s2, m2, d2))

        s_conn = jnp.concatenate([t[0] for t in terms], axis=-2)
        mels = jnp.concatenate([t[1] for t in terms], axis=-1)
        diag = sum(t[2] for t in terms)
        return s_conn, mels, diag
