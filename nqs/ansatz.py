"""Neural-network ansätze for spin-1/2 lattice systems.

Each ansatz is an ``flax.linen.Module`` that maps a configuration
``s ∈ {-1, +1}^{B × N}`` to a complex log-amplitude
``log ψ(s) = a(s) + i ϕ(s)`` of shape ``(B,)``.

Provided architectures:
    * ``CNN``  — periodic-padded deep convolutional network with two
      real heads (log-amplitude and phase). Naturally translation
      equivariant before the global pooling.
    * ``GCNN`` — group-equivariant CNN over the full space group
      ``T × C4v`` of the square lattice (Roth & MacDonald, PRB 2021;
      Roth, Szabó, MacDonald, PRB 2023). Features live on group
      elements and are mixed by true group convolutions; final
      character projection enforces invariance under the chosen irrep.
    * ``ViT``  — Vision-Transformer ansatz of Viteritti, Rende, Becca
      (PRL 2023): patch-embed the spin lattice, run a stack of
      multi-head self-attention blocks, sum-pool to a scalar.
"""

from __future__ import annotations

from typing import Sequence
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from .groups import SpaceGroup


# ----------------------------------------------------------------------------- #
#  utilities                                                                    #
# ----------------------------------------------------------------------------- #


def _periodic_pad_2d(x: jnp.ndarray, pad: int) -> jnp.ndarray:
    """Periodic padding for ``(B, H, W, C)`` tensors."""
    if pad == 0:
        return x
    x = jnp.concatenate([x[:, -pad:], x, x[:, :pad]], axis=1)
    x = jnp.concatenate([x[:, :, -pad:], x, x[:, :, :pad]], axis=2)
    return x


def _periodic_pad_1d(x: jnp.ndarray, pad: int) -> jnp.ndarray:
    if pad == 0:
        return x
    return jnp.concatenate([x[:, -pad:], x, x[:, :pad]], axis=1)


# ----------------------------------------------------------------------------- #
#  CNN ansatz                                                                   #
# ----------------------------------------------------------------------------- #


class CNN(nn.Module):
    """Translationally-equivariant CNN with two real heads.

    Args
    ----
    shape         : ``(L,)`` for 1D, ``(L, L)`` for 2D
    features      : tuple of channel widths
    kernel_size   : odd integer
    """

    shape: Sequence[int]
    features: Sequence[int] = (16, 16, 16)
    kernel_size: int = 3
    activation: str = "gelu"

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:  # (B,) complex
        act = {"gelu": nn.gelu, "tanh": jnp.tanh, "silu": nn.silu}[self.activation]
        ndim = len(self.shape)
        assert ndim in (1, 2)

        x = s.reshape((s.shape[0], *self.shape, 1)).astype(jnp.float32)
        pad = self.kernel_size // 2

        for c in self.features:
            if ndim == 2:
                x = _periodic_pad_2d(x, pad)
                x = nn.Conv(c, (self.kernel_size, self.kernel_size), padding="VALID",
                            kernel_init=nn.initializers.lecun_normal())(x)
            else:
                x = _periodic_pad_1d(x, pad)
                x = nn.Conv(c, (self.kernel_size,), padding="VALID",
                            kernel_init=nn.initializers.lecun_normal())(x)
            x = act(x)

        # Two scalar heads: log|ψ| and phase.
        # Pool by sum (translation invariant after periodic conv).
        pooled = jnp.sum(x, axis=tuple(range(1, ndim + 1)))  # (B, C)
        amp = nn.Dense(1, kernel_init=nn.initializers.lecun_normal())(pooled)[..., 0]
        phase = nn.Dense(1, kernel_init=nn.initializers.zeros)(pooled)[..., 0]
        return amp + 1j * phase


# ----------------------------------------------------------------------------- #
#  Group-equivariant CNN  (GCNN)                                                #
# ----------------------------------------------------------------------------- #


class GCNN(nn.Module):
    """Group-equivariant CNN ansatz on the 2D square lattice.

    Architecture (Roth & MacDonald, PRB 104, 195104, 2021):

        * **Lift**: feature map ``f^{(0)}_g[c] = W_c · (s permuted by g)``
          for each space-group element ``g`` and channel ``c``.
        * **Group convolution(s)**:
            ``f^{(ℓ+1)}_g[c'] = σ( Σ_{h, c} K^{(ℓ)}_{gh, c, c'} f^{(ℓ)}_h[c] )``
          implemented in the regular representation by gather
          + tensor contraction with a learnable kernel ``K``.
        * **Pooling / character projection**:
            ``log ψ(s) = (1/|G|) Σ_g χ̄(g) · h(f^{(L)}_g)``
          where ``χ`` is the character of the chosen irrep
          (default: trivial irrep of ``T × C4v`` ⇒ A₁, ground state).

    The output is complex via two real readout heads as in :class:`CNN`.

    Parameters
    ----------
    L            : side of the square lattice (``L × L`` sites)
    point_group  : ``'C1' | 'C4' | 'C4v'`` (default ``'C4v'``)
    features     : tuple of channel widths
    activation   : ``'gelu' | 'tanh' | 'silu'``
    """

    L: int
    point_group: str = "C4v"
    features: Sequence[int] = (8, 8, 8)
    activation: str = "gelu"

    def setup(self):
        sg = SpaceGroup(L=self.L, point_group=self.point_group)
        perms, mult, inv = sg.build()
        self._perms = jnp.asarray(perms)        # (G, N)
        self._mult = jnp.asarray(mult)          # (G, G)
        self._inv = jnp.asarray(inv)            # (G,)

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        act = {"gelu": nn.gelu, "tanh": jnp.tanh, "silu": nn.silu}[self.activation]
        B = s.shape[0]
        G, N = self._perms.shape

        # ---- lift: (B, N) → (B, G, C0) ----
        s_g = s[:, self._perms].astype(jnp.float32)        # (B, G, N)
        x = nn.Dense(
            self.features[0],
            kernel_init=nn.initializers.lecun_normal(),
            name="lift",
        )(s_g)                                              # (B, G, C0)
        x = act(x)

        # ---- group convolutions over the regular representation ----
        # Pre-compute the index table  T[g, h] = g · h⁻¹  so that a group
        # conv reads as
        #     y_g[c'] = Σ_{h, c} K_{T[g,h], c, c'} · x_h[c]
        # i.e. a plain tensordot after a single `gather` along axis G.
        T = self._mult[:, self._inv]                         # (G, G), int

        for ell, c in enumerate(self.features[1:]):
            kernel = self.param(
                f"gconv_K_{ell}",
                nn.initializers.lecun_normal(),
                (G, x.shape[-1], c),
            )
            # Gather K rows:  Kg[g, h, c_in, c_out] = kernel[T[g,h]]
            # That tensor is (G, G, C_in, C_out); contract h, c_in with x_h.
            Kg = kernel[T]                                   # (G, G, C_in, C_out)
            x = jnp.einsum("bhc,ghcd->bgd", x, Kg) / jnp.sqrt(G)
            x = act(x)

        # ---- character projection (trivial irrep ⇒ uniform sum) ----
        chi = jnp.ones((G,), dtype=x.dtype) / G
        pooled = jnp.einsum("g,bgc->bc", chi, x)             # (B, C)

        amp = nn.Dense(1, kernel_init=nn.initializers.lecun_normal(),
                       name="head_amp")(pooled)[..., 0]
        phase = nn.Dense(1, kernel_init=nn.initializers.zeros,
                         name="head_phase")(pooled)[..., 0]
        return amp + 1j * phase


# ----------------------------------------------------------------------------- #
#  Vision-Transformer ansatz                                                    #
# ----------------------------------------------------------------------------- #


class _MHSA(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x):  # x: (B, T, D)
        return nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x, x)


class _Block(nn.Module):
    d_model: int
    n_heads: int
    mlp_mult: int = 4

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        x = x + _MHSA(self.d_model, self.n_heads)(h)
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_mult * self.d_model)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return x + h


class ViT(nn.Module):
    """Vision-Transformer ansatz for 2D spin lattices.

    Patches the ``L × L`` lattice into ``(L/p) × (L/p)`` tokens of size
    ``p × p``, embeds each patch with a learned linear map plus a
    learned positional embedding, applies ``n_layers`` self-attention
    blocks, sum-pools the tokens, and produces two real heads
    ``log|ψ|`` and ``phase``.
    """

    L: int
    patch: int = 2
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        assert self.L % self.patch == 0, "patch size must divide L"
        B = s.shape[0]
        L, p = self.L, self.patch
        nP = L // p
        T = nP * nP

        x = s.reshape(B, L, L, 1).astype(jnp.float32)
        # Cut into (B, nP, nP, p*p)
        x = x.reshape(B, nP, p, nP, p, 1)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, T, p * p)

        x = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform())(x)
        pos = self.param("pos", nn.initializers.normal(0.02), (T, self.d_model))
        x = x + pos

        for _ in range(self.n_layers):
            x = _Block(self.d_model, self.n_heads)(x)

        x = nn.LayerNorm()(x)
        pooled = jnp.sum(x, axis=1)  # (B, d_model)
        amp = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(pooled)[..., 0]
        phase = nn.Dense(1, kernel_init=nn.initializers.zeros)(pooled)[..., 0]
        return amp + 1j * phase
