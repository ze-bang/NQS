# NQS — Neural Quantum States for Lattice Spin Systems

A compact, state-of-the-art implementation of variational neural quantum
states (NQS) for spin-1/2 lattice models, written in JAX + Flax.

## What's inside

| Component | Notes |
|---|---|
| **Ansätze** | Complex-valued deep CNN (translationally equivariant), **GCNN** — a true group-equivariant CNN over the full `T × C4v` space group (Roth & MacDonald, *PRB* 2021), and a Vision-Transformer ansatz à la *Viteritti, Rende, Becca, PRL 2023*. All output `log ψ = a(s) + i ϕ(s)`. |
| **Optimizer** | **MinSR** — the kernel-trick reformulation of Stochastic Reconfiguration (Chen & Heyl, *Nat. Phys.* 2024), scaling as `O(N_s² P)` instead of `O(P³)`. Falls back to AdamW for cheap pre-training. |
| **Sampler** | Vectorised Metropolis — single-spin flip (TFIM) and sublattice exchange (Heisenberg, conserves total `Sᶻ`), with parallel chains and chunking. |
| **Hamiltonians** | Transverse-Field Ising and (frustrated) Heisenberg J₁–J₂ on 1D chains and 2D square lattices with periodic boundary conditions. Marshall sign rule supported. |
| **Symmetries** | Lattice translation and `Z₂` spin-flip projection at evaluation time (configurable). |
| **Validation** | Exact diagonalisation reference for `N ≤ 20` for unit testing. |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

JAX is installed CPU-only by default; for GPU / TPU follow the
[JAX install guide](https://github.com/google/jax#installation).

## Quick start

```bash
python examples/tfim_1d.py             # 1D TFIM,  CNN ansatz
python examples/heisenberg_2d.py       # 4x4 Heisenberg, ViT ansatz
python examples/heisenberg_2d_gcnn.py  # 4x4 Heisenberg, GCNN ansatz (best)
```

### Reproducible numbers (CPU, no GPU required)

| Example | Lattice | Ansatz | Params | Steps | NQS energy | Exact (ED) | Rel. error |
|---|---|---|---|---|---|---|---|
| `tfim_1d.py` | 1D L = 8, h/J = 1 | CNN | 250 | 150 | −10.2503 | −10.2517 | 1.3 × 10⁻⁴ |
| `heisenberg_2d.py` | 4 × 4, J₂ = 0 | ViT (d = 16, 2 layers) | 6.7 k | 120 | −11.130 | −11.228 | 8.8 × 10⁻³ |
| `heisenberg_2d_gcnn.py` | 4 × 4, J₂ = 0 | **GCNN** `T × C4v`, (6, 6) | **4.7 k** | **80** | **−11.2275** | −11.228 | **9 × 10⁻⁵** |

The GCNN is ~100× more accurate than the ViT on this test with **fewer
parameters and fewer training steps**, exactly because the symmetry is
hard-baked into the architecture rather than learned. On a GPU,
widening the GCNN to `features=(16, 16, 16)` typically pushes the 4×4
relative error below 10⁻⁵.

## Layout

```
nqs/
  lattices.py     # Chain1D, Square2D, neighbour bonds, sublattices
  groups.py       # space group T × {C1, C4, C4v}, Cayley table
  hamiltonians.py # TFIM, HeisenbergJ1J2 (sparse connection lists)
  ansatz.py       # CNN, GCNN, ViT  (complex log-amplitude)
  sampler.py      # Metropolis local-flip & exchange
  vmc.py          # local-energy estimator + jacobian
  optimizer.py    # MinSR (kernel SR), simple Adam wrapper
  symmetries.py   # translation / Z2 projection (post-hoc, optional)
  train.py        # training driver with running diagnostics
  ed.py           # exact diagonalisation (small systems)
examples/
  tfim_1d.py
  heisenberg_2d.py
  heisenberg_2d_gcnn.py
```

## References

- *G. Carleo & M. Troyer*, Solving the quantum many-body problem with
  artificial neural networks. **Science** 355, 602 (2017).
- *C. Roth & A.H. MacDonald*, Group convolutional neural networks
  improve quantum state accuracy. **PRB** 104, 195104 (2021).
- *C. Roth, A. Szabó, A.H. MacDonald*, High-accuracy variational Monte
  Carlo for frustrated magnets with deep neural networks. **PRB** 108,
  054410 (2023).
- *L.L. Viteritti, R. Rende, F. Becca*, Transformer variational wave
  functions for frustrated quantum spin systems. **PRL** 130, 236401 (2023).
- *A. Chen & M. Heyl*, Empowering deep neural quantum states through
  efficient optimization. **Nat. Phys.** 20, 1476 (2024).
- *S. Sorella*, Stochastic Reconfiguration for the optimisation of
  variational wave functions. **PRB** 64, 024512 (2001).

## License

MIT.
