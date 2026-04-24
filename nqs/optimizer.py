"""MinSR-based parameter update wrapped in a small dataclass.

Math is in :func:`nqs.vmc.minsr_update`. This module just hosts the
hyper-parameter container, applies the update, and decays the diagonal
shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import optax

from .vmc import local_energy, minsr_update


@dataclass
class MinSRState:
    lr: float = 0.03
    diag_shift: float = 1e-2
    diag_shift_min: float = 1e-4
    diag_shift_decay: float = 1.0


def minsr_step(apply_fn, params, hamiltonian, samples, state: MinSRState):
    """One natural-gradient step. Returns ``(new_params, e, e_err, |Δθ|)``
    and updates ``state.diag_shift`` in place."""

    def core(p, s):
        return minsr_update(apply_fn, p, hamiltonian, s, state.diag_shift)

    e, e_err, update = core(params, samples)
    new_params = jax.tree_util.tree_map(lambda p, u: p - state.lr * u,
                                        params, update)
    norm = jnp.sqrt(sum(jnp.sum(u * u) for u in jax.tree_util.tree_leaves(update)))
    norm = state.lr * norm
    state.diag_shift = max(state.diag_shift * state.diag_shift_decay,
                           state.diag_shift_min)
    return new_params, e, e_err, norm


# --------------------------------------------------------------------------- #
#  Adam pre-trainer (cheap, optional warm-start)                              #
# --------------------------------------------------------------------------- #


def adam_step(apply_fn, params, hamiltonian, samples, opt_state, optimiser):
    eloc, _ = local_energy(apply_fn, params, hamiltonian, samples)
    eloc = jax.lax.stop_gradient(eloc)
    e_mean = jnp.mean(jnp.real(eloc))

    def loss(p):
        log_psi = apply_fn(p, samples)
        return 2.0 * jnp.mean(jnp.real((eloc - e_mean) * jnp.conj(log_psi)))

    grads = jax.grad(loss)(params)
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, e_mean
