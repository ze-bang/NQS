"""2D Heisenberg model on a 4×4 lattice — ViT ansatz with MinSR.

Heisenberg J1 = 1 (J2 = 0 here, but you can frustrate it). Marshall
sign rule is enabled, so the ground state can be represented with a
real-positive amplitude. We restrict to the Sᶻ=0 sector with the
exchange sampler.

Compares the converged variational energy against exact diagonalisation.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from nqs.lattices import Square2D
from nqs.hamiltonians import HeisenbergJ1J2
from nqs.ansatz import ViT
from nqs.sampler import MetropolisExchange
from nqs.train import TrainConfig, train
from nqs.ed import heis_hamiltonian, ground_state


def main():
    L = 4
    lat = Square2D(L)
    J1, J2 = 1.0, 0.0

    H = HeisenbergJ1J2(
        bonds_nn=lat.bonds_nn,
        bonds_nnn=lat.bonds_nnn,
        n_sites=lat.n_sites,
        sublattice=lat.sublattice,
        J1=J1, J2=J2,
        marshall_sign=True,
    )

    # ED reference (note: ED uses *unrotated* Hamiltonian; both must use the
    # same Marshall convention for energies to match. We compare without it.)
    H_ed = heis_hamiltonian(lat.bonds_nn, lat.bonds_nnn, lat.n_sites,
                            J1=J1, J2=J2, marshall=False)
    e0_ed, _ = ground_state(H_ed)
    print(f"[ED] E0 = {e0_ed:.6f}, E0/N = {e0_ed / lat.n_sites:.6f}")

    model = ViT(L=L, patch=2, d_model=16, n_heads=2, n_layers=2)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, lat.n_sites)))
    n_par = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[NQS] ViT parameters: {n_par}")
    apply_fn = model.apply

    sampler = MetropolisExchange(
        n_sites=lat.n_sites, n_chains=32,
        bonds=lat.bonds_nn, n_thermal=50, n_sweeps=1,
    )

    # NOTE: a few hundred steps with n_samples=512 already gets within ~1%
    # of the ED energy on a GPU. The defaults below keep the example light
    # enough to finish on a CPU laptop in ~2 min — feel free to scale up.
    cfg = TrainConfig(
        n_steps=120, n_samples=256, lr=0.03,
        diag_shift=1e-2, diag_shift_min=1e-4, diag_shift_decay=0.98,
        log_every=10, seed=1,
    )
    params, hist = train(apply_fn, params, H, sampler, cfg)

    e_final = np.mean([h[1] for h in hist[-20:]])
    print(f"\n[NQS] mean E (last 20 steps) = {e_final:.6f}")
    print(f"[NQS] relative error          = {(e_final - e0_ed) / abs(e0_ed):.2e}")


if __name__ == "__main__":
    main()
