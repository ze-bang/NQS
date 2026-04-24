"""Training driver: combine sampler + ansatz + Hamiltonian + MinSR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np

from .vmc import minsr_update
from .optimizer import MinSRState


@dataclass
class TrainConfig:
    n_steps: int = 200
    n_samples: int = 1024
    lr: float = 0.03
    diag_shift: float = 1e-2
    diag_shift_min: float = 1e-4
    diag_shift_decay: float = 0.995
    log_every: int = 10
    seed: int = 0


def train(
    apply_fn: Callable,
    params,
    hamiltonian,
    sampler,
    cfg: TrainConfig,
    *,
    init_state=None,
    callback: Optional[Callable] = None,
):
    key = jax.random.PRNGKey(cfg.seed)
    if init_state is None:
        key, sub = jax.random.split(key)
        s0 = sampler.init(sub)
    else:
        s0 = init_state

    n_samples = cfg.n_samples

    @partial(jax.jit, static_argnames=("n_samples",))
    def sample_jit(params, s0, key, n_samples):
        return sampler.sample(apply_fn, params, s0, key, n_samples)

    @jax.jit
    def step_jit(params, samples, diag_shift):
        return minsr_update(apply_fn, params, hamiltonian, samples, diag_shift)

    state = MinSRState(
        lr=cfg.lr, diag_shift=cfg.diag_shift,
        diag_shift_min=cfg.diag_shift_min,
        diag_shift_decay=cfg.diag_shift_decay,
    )

    history = []
    t0 = time.time()
    for it in range(cfg.n_steps):
        key, sub = jax.random.split(key)
        samples, acc, s0 = sample_jit(params, s0, sub, n_samples)
        e, e_err, update = step_jit(params, samples, state.diag_shift)

        params = jax.tree_util.tree_map(lambda p, u: p - state.lr * u,
                                         params, update)
        norm = jnp.sqrt(sum(jnp.sum(u * u)
                            for u in jax.tree_util.tree_leaves(update))) * state.lr

        e_f = float(jax.device_get(e))
        e_err_f = float(jax.device_get(e_err))
        acc_f = float(jax.device_get(acc))
        norm_f = float(jax.device_get(norm))
        history.append((it, e_f, e_err_f, acc_f, norm_f))

        # diag-shift schedule
        state.diag_shift = max(state.diag_shift * state.diag_shift_decay,
                               state.diag_shift_min)

        if (it + 1) % cfg.log_every == 0 or it == 0:
            print(
                f"step {it + 1:4d} | E = {e_f: .6f} ± {e_err_f:.1e} | "
                f"acc = {acc_f:.2f} | δθ = {norm_f:.2e} | "
                f"λ = {state.diag_shift:.1e} | t = {time.time() - t0:5.1f}s",
                flush=True,
            )
        if callback is not None:
            callback(it, e_f, params, state)

    return params, np.array(history, dtype=object)
