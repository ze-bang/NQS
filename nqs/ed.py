"""Tiny exact-diagonalisation reference for spin-1/2 lattice models.

Builds the Hamiltonian as a SciPy sparse matrix; usable up to ``N ≈ 22``.
Used to validate the NQS code on small systems.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


def _basis_states(N):
    return np.arange(1 << N, dtype=np.int64)


def _spins(states, N):
    """States → ±1 spins, shape (D, N), site 0 is the *highest* bit
    (bit=0 → +1, bit=1 → −1)."""
    bits = ((states[:, None] >> np.arange(N - 1, -1, -1)) & 1)
    return 1 - 2 * bits


def _flip(states, i, N):
    return states ^ (1 << (N - 1 - i))


def tfim_hamiltonian(bonds, N, J=1.0, h=1.0):
    from scipy.sparse import coo_matrix
    D = 1 << N
    states = _basis_states(N)
    spins = _spins(states, N)
    diag = -J * np.sum(spins[:, bonds[:, 0]] * spins[:, bonds[:, 1]], axis=1)
    rows = [states]
    cols = [states]
    data = [diag.astype(np.float64)]
    for i in range(N):
        flipped = _flip(states, i, N)
        rows.append(states)
        cols.append(flipped)
        data.append(np.full(D, -h, dtype=np.float64))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    return coo_matrix((data, (rows, cols)), shape=(D, D)).tocsr()


def heis_hamiltonian(bonds_nn, bonds_nnn, N, J1=1.0, J2=0.0, marshall=False):
    from scipy.sparse import coo_matrix
    D = 1 << N
    states = _basis_states(N)
    spins = _spins(states, N)
    rows, cols, data = [], [], []

    def add(bonds, J):
        if J == 0.0 or len(bonds) == 0:
            return
        diag = J * 0.25 * np.sum(spins[:, bonds[:, 0]] * spins[:, bonds[:, 1]], axis=1)
        rows.append(states)
        cols.append(states)
        data.append(diag.astype(np.float64))
        sgn = -1.0 if marshall else 1.0
        for (i, j) in bonds:
            si = spins[:, i]
            sj = spins[:, j]
            antipar = si * sj < 0
            from_ = states[antipar]
            new = _flip(_flip(from_, i, N), j, N)
            rows.append(from_)
            cols.append(new)
            data.append(np.full(from_.shape, J * 0.5 * sgn, dtype=np.float64))

    add(bonds_nn, J1)
    add(bonds_nnn, J2)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    return coo_matrix((data, (rows, cols)), shape=(D, D)).tocsr()


def ground_state(H, k: int = 1):
    """Lowest eigen-pair via Lanczos."""
    H = csr_matrix(H)
    w, v = eigsh(H, k=k, which="SA")
    order = np.argsort(w)
    return float(w[order[0]]), v[:, order[0]]
