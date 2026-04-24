"""Variational Monte Carlo: local energy and the MinSR natural-gradient.

Given an ansatz returning ``log ψ`` on a batch of configurations,

    E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩  ≈  ⟨ E_loc(s) ⟩_{s ~ |ψ|²} ,
    E_loc(s) = Σ_{s'} ⟨s|H|s'⟩  exp(log ψ(s') − log ψ(s)) .

Optimisation needs the per-sample log-derivatives
``O_{ki} = ∂_{θ_i} log ψ(s_k)``. With real parameters and complex
``log ψ``, ``O`` is complex; we compute it via ``jax.vmap(jax.grad(...))``
on the real / imag parts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
#  local energy                                                                #
# --------------------------------------------------------------------------- #


def local_energy(apply_fn, params, hamiltonian, s):
    """``E_loc(s)`` for each row of ``s``."""
    s_conn, mels, diag = hamiltonian.local_terms(s)         # (B,M,N), (B,M), (B,)
    log_psi_s = apply_fn(params, s)                          # (B,)
    B, M, N = s_conn.shape
    log_psi_conn = apply_fn(params, s_conn.reshape(B * M, N)).reshape(B, M)
    ratios = jnp.exp(log_psi_conn - log_psi_s[:, None])
    eloc = diag.astype(ratios.dtype) + jnp.sum(mels.astype(ratios.dtype) * ratios,
                                               axis=-1)
    return eloc, log_psi_s


# --------------------------------------------------------------------------- #
#  jacobian (per-sample log-derivatives)                                       #
# --------------------------------------------------------------------------- #


def _flatten_params(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves]
    sizes = [int(jnp.size(l)) for l in leaves]
    flat = jnp.concatenate([l.reshape(-1) for l in leaves])
    return flat, treedef, shapes, sizes


def _unflatten_params(flat, treedef, shapes, sizes):
    leaves, off = [], 0
    for sz, sh in zip(sizes, shapes):
        leaves.append(flat[off:off + sz].reshape(sh))
        off += sz
    return jax.tree_util.tree_unflatten(treedef, leaves)


def jacobian_log_psi(apply_fn, params, s):
    """Return complex ``(B, P)`` Jacobian of ``log ψ`` w.r.t. flat real
    parameters."""
    flat, treedef, shapes, sizes = _flatten_params(params)

    def log_psi_single(p_flat, s_single):
        p = _unflatten_params(p_flat, treedef, shapes, sizes)
        return apply_fn(p, s_single[None, :])[0]

    real = jax.vmap(jax.grad(lambda p, x: jnp.real(log_psi_single(p, x))),
                    in_axes=(None, 0))(flat, s)
    imag = jax.vmap(jax.grad(lambda p, x: jnp.imag(log_psi_single(p, x))),
                    in_axes=(None, 0))(flat, s)
    return real + 1j * imag, treedef, shapes, sizes


# --------------------------------------------------------------------------- #
#  MinSR core                                                                 #
# --------------------------------------------------------------------------- #


def minsr_update(apply_fn, params, hamiltonian, s, diag_shift):
    """Compute the natural-gradient parameter update via MinSR.

    Returns ``(e_mean, e_err, update_pytree)`` where ``update_pytree`` has
    the same structure as ``params`` and contains the *unscaled* SR
    direction (multiply by the learning rate to apply).
    """
    eloc, _ = local_energy(apply_fn, params, hamiltonian, s)
    e_mean = jnp.mean(jnp.real(eloc))
    e_err = jnp.sqrt(jnp.var(jnp.real(eloc)) / s.shape[0])
    eps = eloc - jnp.mean(eloc)                              # (Ns,) complex

    O, treedef, shapes, sizes = jacobian_log_psi(apply_fn, params, s)
    O_c = O - jnp.mean(O, axis=0, keepdims=True)             # (Ns, P) complex

    Ns = s.shape[0]
    O_real = jnp.concatenate([O_c.real, O_c.imag], axis=0)   # (2Ns, P)
    eps_real = jnp.concatenate([eps.real, eps.imag], axis=0) # (2Ns,)

    T = (O_real @ O_real.T) / Ns                             # (2Ns, 2Ns)
    rhs = eps_real / Ns
    v = jnp.linalg.solve(T + diag_shift * jnp.eye(2 * Ns, dtype=T.dtype), rhs)
    dtheta_flat = O_real.T @ v                                # (P,) real

    update = _unflatten_params(dtheta_flat, treedef, shapes, sizes)
    return e_mean, e_err, update
