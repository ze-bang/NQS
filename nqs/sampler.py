"""Vectorised Metropolis–Hastings samplers.

Two move kernels are provided:

* ``MetropolisLocal`` proposes a single-spin flip per step. Suitable
  for non-conserving Hamiltonians (e.g. TFIM).
* ``MetropolisExchange`` proposes swapping two opposite spins on a
  random nearest-neighbour bond. Conserves total ``Sᶻ`` — the right
  kernel for the Heisenberg model when working in a fixed magnetisation
  sector (typically ``Sᶻ_total = 0``).

Both kernels run ``n_chains`` chains in parallel, fully ``jit``-able.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np


def _log_psi_batched(apply_fn, params, s):
    return apply_fn(params, s)


@dataclass
class MetropolisLocal:
    n_sites: int
    n_chains: int
    n_sweeps: int = 2  # one "sweep" = n_sites proposals
    n_thermal: int = 50

    def init(self, key):
        s = jax.random.choice(
            key, jnp.array([-1.0, 1.0]), shape=(self.n_chains, self.n_sites)
        )
        return s

    def _step(self, apply_fn, params, s, key):
        kf, ka = jax.random.split(key)
        flip_idx = jax.random.randint(kf, (self.n_chains,), 0, self.n_sites)
        ones = jnp.ones((self.n_chains, self.n_sites), dtype=s.dtype)
        flips = ones.at[jnp.arange(self.n_chains), flip_idx].set(-1.0)
        s_new = s * flips
        log_psi_old = apply_fn(params, s)
        log_psi_new = apply_fn(params, s_new)
        log_p = 2.0 * jnp.real(log_psi_new - log_psi_old)
        u = jax.random.uniform(ka, (self.n_chains,))
        accept = jnp.log(u) < log_p
        s_out = jnp.where(accept[:, None], s_new, s)
        return s_out, accept.mean()

    def sample(self, apply_fn, params, s0, key, n_samples):
        """Run chains, returning ``n_samples`` configurations and the
        mean acceptance rate over the production phase."""

        def body(carry, key):
            s, _ = carry
            s, acc = self._step(apply_fn, params, s, key)
            return (s, acc), acc

        # Thermalise.
        keys_th = jax.random.split(key, self.n_thermal * self.n_sites)
        (s, _), _ = jax.lax.scan(body, (s0, 0.0), keys_th)

        # Production: collect ``n_samples / n_chains`` samples per chain,
        # decorrelated by ``n_sweeps * n_sites`` proposals.
        per_chain = (n_samples + self.n_chains - 1) // self.n_chains
        samples = []
        accs = []
        decorr = self.n_sweeps * self.n_sites
        keys = jax.random.split(jax.random.fold_in(key, 1), per_chain * decorr)
        keys = keys.reshape(per_chain, decorr, 2)

        def collect(carry, key_block):
            s, _ = carry
            (s, acc), _ = jax.lax.scan(body, (s, 0.0), key_block)
            return (s, acc), (s, acc)

        (s, _), (samples, accs) = jax.lax.scan(collect, (s, 0.0), keys)
        # samples: (per_chain, n_chains, N) → (n_samples, N)
        samples = samples.reshape(-1, self.n_sites)[:n_samples]
        return samples, jnp.mean(accs), s


@dataclass
class MetropolisExchange:
    """Exchange update on bonds: pick a random bond, swap if antiparallel."""

    n_sites: int
    n_chains: int
    bonds: np.ndarray  # (E, 2)
    n_sweeps: int = 2
    n_thermal: int = 50

    def init(self, key, total_sz: int = 0):
        """Initialise chains with given total ``Sᶻ`` (in units of σᶻ sums)."""
        n_up = (self.n_sites + total_sz) // 2
        assert 0 <= n_up <= self.n_sites
        keys = jax.random.split(key, self.n_chains)

        def one(k):
            base = jnp.concatenate([jnp.ones(n_up), -jnp.ones(self.n_sites - n_up)])
            return jax.random.permutation(k, base)

        return jax.vmap(one)(keys)

    def _step(self, apply_fn, params, s, key):
        kb, ka = jax.random.split(key)
        bonds = jnp.asarray(self.bonds)
        E = bonds.shape[0]
        bidx = jax.random.randint(kb, (self.n_chains,), 0, E)
        i = bonds[bidx, 0]
        j = bonds[bidx, 1]
        si = s[jnp.arange(self.n_chains), i]
        sj = s[jnp.arange(self.n_chains), j]
        antipar = si * sj < 0
        # Build s_new by swapping i and j on antiparallel chains.
        s_new = s.at[jnp.arange(self.n_chains), i].set(sj)
        s_new = s_new.at[jnp.arange(self.n_chains), j].set(si)
        s_new = jnp.where(antipar[:, None], s_new, s)
        log_psi_old = apply_fn(params, s)
        log_psi_new = apply_fn(params, s_new)
        log_p = 2.0 * jnp.real(log_psi_new - log_psi_old)
        u = jax.random.uniform(ka, (self.n_chains,))
        accept = (jnp.log(u) < log_p) & antipar
        s_out = jnp.where(accept[:, None], s_new, s)
        return s_out, accept.mean()

    def sample(self, apply_fn, params, s0, key, n_samples):
        def body(carry, key):
            s, _ = carry
            s, acc = self._step(apply_fn, params, s, key)
            return (s, acc), acc

        keys_th = jax.random.split(key, self.n_thermal * self.n_sites)
        (s, _), _ = jax.lax.scan(body, (s0, 0.0), keys_th)

        per_chain = (n_samples + self.n_chains - 1) // self.n_chains
        decorr = self.n_sweeps * self.n_sites
        keys = jax.random.split(jax.random.fold_in(key, 1), per_chain * decorr)
        keys = keys.reshape(per_chain, decorr, 2)

        def collect(carry, key_block):
            s, _ = carry
            (s, acc), _ = jax.lax.scan(body, (s, 0.0), key_block)
            return (s, acc), (s, acc)

        (s, _), (samples, accs) = jax.lax.scan(collect, (s, 0.0), keys)
        samples = samples.reshape(-1, self.n_sites)[:n_samples]
        return samples, jnp.mean(accs), s
