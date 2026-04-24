"""Lattice geometries.

Each lattice exposes:
    n_sites          : int       — number of spins
    shape            : tuple     — natural tensor shape (e.g. (L,) or (L, L))
    bonds_nn         : (E, 2) int array of nearest-neighbour bonds
    bonds_nnn        : (E', 2) int array of next-nearest-neighbour bonds
    sublattice       : (n_sites,) int array of bipartition indices (0/1)
                       — well defined for bipartite lattices
    translations     : (n_T, n_sites) int permutation array of all
                       lattice translations (used by symmetry projection)
    reshape(s)       : reshape a flat configuration into ``shape``
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Chain1D:
    """Periodic 1D chain of ``L`` sites."""

    L: int

    @property
    def n_sites(self) -> int:
        return self.L

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.L,)

    @property
    def bonds_nn(self) -> np.ndarray:
        L = self.L
        return np.stack([np.arange(L), (np.arange(L) + 1) % L], axis=1)

    @property
    def bonds_nnn(self) -> np.ndarray:
        L = self.L
        return np.stack([np.arange(L), (np.arange(L) + 2) % L], axis=1)

    @property
    def sublattice(self) -> np.ndarray:
        return np.arange(self.L) % 2

    @property
    def translations(self) -> np.ndarray:
        L = self.L
        idx = np.arange(L)
        return np.stack([(idx + t) % L for t in range(L)], axis=0)

    def reshape(self, s: np.ndarray) -> np.ndarray:
        return s.reshape(*s.shape[:-1], self.L)


@dataclass(frozen=True)
class Square2D:
    """Periodic L × L square lattice."""

    L: int

    @property
    def n_sites(self) -> int:
        return self.L * self.L

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.L, self.L)

    def _idx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x % self.L) * self.L + (y % self.L)

    @property
    def bonds_nn(self) -> np.ndarray:
        L = self.L
        x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        x, y = x.ravel(), y.ravel()
        i = self._idx(x, y)
        right = self._idx(x + 1, y)
        down = self._idx(x, y + 1)
        return np.concatenate(
            [np.stack([i, right], axis=1), np.stack([i, down], axis=1)], axis=0
        )

    @property
    def bonds_nnn(self) -> np.ndarray:
        L = self.L
        x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        x, y = x.ravel(), y.ravel()
        i = self._idx(x, y)
        d1 = self._idx(x + 1, y + 1)
        d2 = self._idx(x + 1, y - 1)
        return np.concatenate(
            [np.stack([i, d1], axis=1), np.stack([i, d2], axis=1)], axis=0
        )

    @property
    def sublattice(self) -> np.ndarray:
        L = self.L
        x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        return ((x + y) % 2).ravel()

    @property
    def translations(self) -> np.ndarray:
        L = self.L
        N = L * L
        base = np.arange(N).reshape(L, L)
        out = []
        for dx in range(L):
            for dy in range(L):
                out.append(np.roll(base, shift=(dx, dy), axis=(0, 1)).ravel())
        return np.stack(out, axis=0)

    def reshape(self, s: np.ndarray) -> np.ndarray:
        return s.reshape(*s.shape[:-1], self.L, self.L)
