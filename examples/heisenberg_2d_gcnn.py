"""2D Heisenberg model on a 4×4 lattice — GCNN ansatz with MinSR.

Same Hamiltonian as ``heisenberg_2d.py``, but with a group-equivariant
CNN over the full ``T × C4v`` space group (Roth & MacDonald, PRB 2021).
Because the GCNN bakes the symmetry into the architecture, it
typically reaches a much better energy with fewer parameters and
fewer optimisation steps than the plain CNN/ViT.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from nqs.lattices import Square2D
from nqs.hamiltonians import HeisenbergJ1J2
from nqs.ansatz import GCNN
from nqs.sampler import MetropolisExchange
from nqs.train import TrainConfig, train
from nqs.ed import heis_hamiltonian, ground_state


def main():
    L = 4
    lat = Square2D(L)
    J1, J2 = 1.0, 0.0

    H = HeisenbergJ1J2(
        bonds_nn=lat.bonds_nn, bonds_nnn=lat.bonds_nnn,
        n_sites=lat.n_sites, sublattice=lat.sublattice,
        J1=J1, J2=J2, marshall_sign=True,
    )

    H_ed = heis_hamiltonian(lat.bonds_nn, lat.bonds_nnn, lat.n_sites,
                            J1=J1, J2=J2, marshall=False)
    e0_ed, _ = ground_state(H_ed)
    print(f"[ED] E0 = {e0_ed:.6f}, E0/N = {e0_ed / lat.n_sites:.6f}")

    model = GCNN(L=L, point_group="C4v", features=(6, 6))
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, lat.n_sites)))
    n_par = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[NQS] GCNN parameters: {n_par}")
    apply_fn = model.apply

    sampler = MetropolisExchange(
        n_sites=lat.n_sites, n_chains=32,
        bonds=lat.bonds_nn, n_thermal=50, n_sweeps=1,
    )

    # Feel free to bump features=(8, 8, 8) and n_steps=200 on a GPU to
    # push the relative error well below 1e-4.
    cfg = TrainConfig(
        n_steps=80, n_samples=256, lr=0.05,
        diag_shift=1e-2, diag_shift_min=1e-5, diag_shift_decay=0.95,
        log_every=10, seed=1,
    )
    params, hist = train(apply_fn, params, H, sampler, cfg)

    e_final = np.mean([h[1] for h in hist[-20:]])
    print(f"\n[NQS] mean E (last 20 steps) = {e_final:.6f}")
    print(f"[NQS] relative error          = {(e_final - e0_ed) / abs(e0_ed):.2e}")


if __name__ == "__main__":
    main()
