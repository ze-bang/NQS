"""Symmetry projection of the wavefunction at evaluation time.

Given a base ansatz ``log ψ_θ(s)`` and a group ``G`` of permutations
acting on lattice sites, a symmetric (trivial-rep) wavefunction is

    ψ_sym(s) = (1/|G|) Σ_{g ∈ G} ψ_θ(g·s) ,

implemented in log-space via ``logsumexp``. Optionally we add a
``Z₂`` spin-flip projection that averages over ``s`` and ``-s``.

Useful as a wrapper at *evaluation* time; for training one usually
trains the bare ansatz with a translation-equivariant body (the CNN
already satisfies this) and adds the projection in post.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def project(apply_fn, perms: jnp.ndarray, *, z2: bool = False):
    """Return a new apply function that projects onto the trivial rep.

    Args
    ----
    apply_fn : function (params, s) → log ψ
    perms    : (G, N) integer permutation array (e.g. lattice translations)
    z2       : also average over global spin flip
    """

    perms = jnp.asarray(perms)

    def projected(params, s):
        # Apply each permutation: (G, B, N)
        s_g = s[None, :, perms]            # (G, B, N)
        G, B, N = s_g.shape
        flat = s_g.reshape(G * B, N)
        log_psi = apply_fn(params, flat).reshape(G, B)
        if z2:
            log_psi_flip = apply_fn(params, -flat).reshape(G, B)
            log_psi = jnp.concatenate([log_psi, log_psi_flip], axis=0)
            G *= 2
        # logsumexp - log G
        return jax.scipy.special.logsumexp(log_psi, axis=0) - jnp.log(G)

    return projected
