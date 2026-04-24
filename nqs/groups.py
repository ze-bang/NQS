"""Lattice space groups and their Cayley tables.

For a 2D periodic ``L × L`` square lattice we expose the full space
group ``G = T × P`` where ``T`` is the translation group (size ``L²``)
and ``P`` is one of

    'C1'  : trivial          (|P| = 1, total |G| = L²)
    'C4'  : 4-fold rotations (|P| = 4, total |G| = 4 L²)
    'C4v' : 4 rotations + 4 reflections (|P| = 8, total |G| = 8 L²)

Each group element is represented as a permutation of the ``N = L²``
sites, so ``perms[g, i]`` is the site that maps to ``i`` under ``g``
(equivalently, applying ``g`` to a configuration ``s`` gives
``s_g[i] = s[perms[g, i]]``). With this convention the composition law
is ``perms[g·h] = perms[h] ∘ perms[g]`` (right-to-left), which we use
to build the Cayley table once at construction time.

The Cayley table ``mult[g, h] = g·h`` and the inverse table ``inv[g]``
are then exported as static integer arrays, ready to drive a true
group convolution over the regular representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------- #
#  point-group action on a 2D coordinate                                     #
# --------------------------------------------------------------------------- #


def _point_group_ops(name: str):
    """Return a list of 2×2 integer matrices acting on (x, y)."""
    R = np.array([[0, -1], [1, 0]])      # 90° rotation
    Mx = np.array([[1, 0], [0, -1]])     # reflection across x-axis
    if name == "C1":
        return [np.eye(2, dtype=int)]
    if name == "C4":
        return [np.linalg.matrix_power(R, k).astype(int) for k in range(4)]
    if name == "C4v":
        rot = [np.linalg.matrix_power(R, k).astype(int) for k in range(4)]
        return rot + [r @ Mx for r in rot]
    raise ValueError(f"unknown point group {name!r}")


# --------------------------------------------------------------------------- #
#  build the space group as permutations of the L² sites                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpaceGroup:
    L: int
    point_group: str = "C4v"

    def build(self):
        """Return ``(perms, mult, inv)`` arrays.

        perms : (G, N) int — permutation representation of each element
        mult  : (G, G) int — Cayley table mult[g, h] = g · h
        inv   : (G,)   int — inverse element of each g
        """
        L, N = self.L, self.L * self.L
        ops = _point_group_ops(self.point_group)

        # Each element = (translation by (tx, ty)) ∘ (point op p)
        # Action on site (x, y): (x', y') = R_p · (x, y) + (tx, ty)  (mod L)
        coords = np.stack(
            np.meshgrid(np.arange(L), np.arange(L), indexing="ij"), axis=-1
        ).reshape(N, 2)  # (N, 2)

        perms = []
        labels = []  # for inverse lookup
        for tx in range(L):
            for ty in range(L):
                for ip, R in enumerate(ops):
                    new = (coords @ R.T + np.array([tx, ty])) % L  # (N, 2)
                    new_idx = new[:, 0] * L + new[:, 1]
                    # perms[g, i] = j  ⇔  applying g sends site j to position i
                    # We define s_g[i] = s[perms[g, i]]; i.e. perms[g] is the
                    # *inverse* permutation of "where does i go under g".
                    # Construct it by inverting new_idx (which is "where does
                    # j go" in functional form).
                    inv_perm = np.empty(N, dtype=np.int64)
                    inv_perm[new_idx] = np.arange(N)
                    perms.append(inv_perm)
                    labels.append((tx, ty, ip))
        perms = np.stack(perms, axis=0)
        G = perms.shape[0]

        # Build Cayley table by matching composed permutations.
        # Composition: s_{gh}[i] = s[perms[gh, i]]
        # Also: s_h applied by g: (s_h)_g[i] = s_h[perms[g, i]] = s[perms[h, perms[g, i]]]
        # Therefore perms[g·h, i] = perms[h, perms[g, i]] = perms[h][perms[g]]
        # Identify each composed permutation against the table.
        # For matching, hash each row to a tuple.
        key_to_index = {tuple(p.tolist()): i for i, p in enumerate(perms)}
        mult = np.empty((G, G), dtype=np.int64)
        for g in range(G):
            composed = perms[:, perms[g]]   # (G, N) — this is perms[h][perms[g]] for each h
            for h in range(G):
                mult[g, h] = key_to_index[tuple(composed[h].tolist())]

        # Inverse: g·g⁻¹ = e (the identity is the row equal to arange(N)).
        ident = np.arange(N, dtype=np.int64)
        e_idx = key_to_index[tuple(ident.tolist())]
        inv = np.empty(G, dtype=np.int64)
        for g in range(G):
            for h in range(G):
                if mult[g, h] == e_idx:
                    inv[g] = h
                    break
        return perms, mult, inv

    @property
    def n_elements(self) -> int:
        L = self.L
        sizes = {"C1": 1, "C4": 4, "C4v": 8}
        return sizes[self.point_group] * L * L
