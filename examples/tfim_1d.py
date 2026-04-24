"""1D Transverse-Field Ising model — NQS vs exact diagonalisation.

Trains a small CNN ansatz on the L=10 critical TFIM (h/J = 1) and
prints the relative error against the exact ground-state energy.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from nqs.lattices import Chain1D
from nqs.hamiltonians import TFIM
from nqs.ansatz import CNN
from nqs.sampler import MetropolisLocal
from nqs.train import TrainConfig, train
from nqs.ed import tfim_hamiltonian, ground_state


def main():
    L = 8
    lat = Chain1D(L)
    H = TFIM(bonds=lat.bonds_nn, n_sites=lat.n_sites, J=1.0, h=1.0)

    H_ed = tfim_hamiltonian(lat.bonds_nn, L, J=1.0, h=1.0)
    e0_ed, _ = ground_state(H_ed)
    print(f"[ED] E0 = {e0_ed:.6f}, E0/N = {e0_ed / L:.6f}")

    model = CNN(shape=lat.shape, features=(8, 8), kernel_size=3)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, lat.n_sites)))
    apply_fn = model.apply

    sampler = MetropolisLocal(n_sites=lat.n_sites, n_chains=32, n_thermal=50,
                              n_sweeps=1)
    cfg = TrainConfig(
        n_steps=150, n_samples=256, lr=0.05,
        diag_shift=1e-2, diag_shift_min=1e-4, diag_shift_decay=0.99,
        log_every=10, seed=1,
    )
    params, hist = train(apply_fn, params, H, sampler, cfg)

    e_final = np.mean([h[1] for h in hist[-20:]])
    print(f"\n[NQS] mean E (last 20 steps) = {e_final:.6f}")
    print(f"[NQS] relative error          = {(e_final - e0_ed) / abs(e0_ed):.2e}")


if __name__ == "__main__":
    main()
