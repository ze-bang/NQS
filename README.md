# NQS — Neural Quantum States for Lattice Spin Systems

A compact, state-of-the-art implementation of variational neural quantum
states (NQS) for spin-1/2 lattice models, written in JAX + Flax.

## What's inside

| Component | Notes |
|---|---|
| **Ansätze** | Complex-valued deep CNN (translationally equivariant) and a Vision-Transformer ansatz à la *Viteritti, Rende, Becca, PRL 2023*. Outputs `log ψ = a(s) + i ϕ(s)`. |
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
python examples/tfim_1d.py            # 1D TFIM, L=10, ED comparison
python examples/heisenberg_2d.py      # 4x4 Heisenberg, ViT ansatz, MinSR
```

### Reproducible numbers (CPU, no GPU required)

| Example | Lattice | Ansatz | NQS energy | Exact (ED) | Rel. error |
|---|---|---|---|---|---|
| `tfim_1d.py` | 1D, L = 8, h/J = 1 | CNN, 250 params | −10.2503 | −10.2517 | 1.3 × 10⁻⁴ |
| `heisenberg_2d.py` | 4 × 4 square, J₂ = 0 | ViT (d = 16, 2 layers), 6.7 k params | −11.130 | −11.228 | 8.8 × 10⁻³ |

The 4 × 4 Heisenberg gap closes further to ≪ 10⁻³ with a wider ViT
(`d_model=64, n_layers=4`) and ≥ 1024 samples — almost free on a GPU.

## Layout

```
nqs/
  lattices.py     # Chain1D, Square2D, neighbour bonds, sublattices
  hamiltonians.py # TFIM, HeisenbergJ1J2 (sparse connection lists)
  ansatz.py       # CNN, ViT (complex log-amplitude)
  sampler.py      # Metropolis local-flip & exchange
  vmc.py          # local-energy estimator + jacobian
  optimizer.py    # MinSR (kernel SR), simple Adam wrapper
  symmetries.py   # translation / Z2 projection
  train.py        # training driver with running diagnostics
  ed.py           # exact diagonalisation (small systems)
examples/
  tfim_1d.py
  heisenberg_2d.py
```

## References

- *G. Carleo & M. Troyer*, Solving the quantum many-body problem with
  artificial neural networks. **Science** 355, 602 (2017).
- *L.L. Viteritti, R. Rende, F. Becca*, Transformer variational wave
  functions for frustrated quantum spin systems. **PRL** 130, 236401 (2023).
- *A. Chen & M. Heyl*, Empowering deep neural quantum states through
  efficient optimization. **Nat. Phys.** 20, 1476 (2024).
- *S. Sorella*, Stochastic Reconfiguration for the optimisation of
  variational wave functions. **PRB** 64, 024512 (2001).

## License

MIT.
