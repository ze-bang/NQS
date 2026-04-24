"""Neural Quantum States for lattice spin systems.

A small JAX/Flax library implementing a modern variational Monte Carlo
stack for spin-1/2 models on regular lattices: complex-valued CNN and
Vision-Transformer ansätze, vectorised Metropolis sampling, and the
MinSR (kernel-trick Stochastic Reconfiguration) optimiser.
"""

from .lattices import Chain1D, Square2D
from .hamiltonians import TFIM, HeisenbergJ1J2
from .ansatz import CNN, GCNN, ViT
from .sampler import MetropolisLocal, MetropolisExchange
from .vmc import local_energy, minsr_update
from .optimizer import MinSRState, minsr_step
from .train import train

__all__ = [
    "Chain1D",
    "Square2D",
    "TFIM",
    "HeisenbergJ1J2",
    "CNN",
    "GCNN",
    "ViT",
    "MetropolisLocal",
    "MetropolisExchange",
    "local_energy",
    "minsr_update",
    "MinSRState",
    "minsr_step",
    "train",
]
