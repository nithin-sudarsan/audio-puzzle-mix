"""Vanilla Mixup augmentation for mel-spectrogram training.

Vanilla Mixup (Zhang et al., 2018) constructs a mixed training example by
taking a convex combination of two input spectrograms and their labels:

    x_mixed = λ · x₁ + (1 − λ) · x₂
    y_mixed = λ · y₁ + (1 − λ) · y₂

where λ ~ Beta(α, α), and (x₂, y₂) is drawn from a random permutation of
the same batch as (x₁, y₁).

Mixing is performed at the *input* level (pixel space), meaning the model
sees a smooth blend of two spectrograms.  This is the simplest form of
mixing — it deliberately ignores any spatial structure.

Why this baseline is informative
---------------------------------
Vanilla Mixup produces mixed examples that are globally ambiguous: every
frequency bin at every time step is a blend of both sources.  There are no
regions that "belong to" one class exclusively.  From an XAI perspective this
is the worst case: if the model learns only from globally blended inputs, its
saliency maps may become diffuse (attending to everything weakly rather than
any feature strongly), because no clear spatial correspondence between input
regions and class identity is reinforced during training.

This is the mechanism we expect to produce the lowest FBC and highest saliency
entropy scores in our evaluation — though confirming or falsifying that
expectation is part of what the experiments test.

Mixing coefficient distribution
---------------------------------
λ ~ Beta(α, α) with α = 0.4.  The Beta(0.4, 0.4) distribution is U-shaped:
it concentrates mass near λ ≈ 0 and λ ≈ 1, meaning most mixed examples are
predominantly one class with a small admixture of the other.  Truly 50/50
blends are rare.  This is intentional: extremely ambiguous examples (λ ≈ 0.5)
are harder to learn from and can harm convergence.

Interface contract
-------------------
All augmentation modules in this project share the same function signature:

    x_mixed, y_mixed = apply_*(x, y_onehot, alpha, ...)

where:
  x         : Tensor (B, 1, H, W)   — batch of spectrograms
  y_onehot  : Tensor (B, num_classes)— one-hot (or already-soft) labels
  alpha     : float                  — Beta distribution parameter
  x_mixed   : Tensor (B, 1, H, W)   — mixed spectrograms
  y_mixed   : Tensor (B, num_classes)— soft mixed labels

This consistent interface means the training loop can call any augmentation
method identically and switch between conditions by changing a single argument.
"""

import numpy as np
import torch
from torch import Tensor


def sample_lambda(alpha: float) -> float:
    """Sample a mixing coefficient λ from Beta(alpha, alpha).

    Beta(α, α) is symmetric around 0.5.  For α < 1 the distribution is
    U-shaped (mass near 0 and 1); for α = 1 it is uniform; for α > 1 it is
    bell-shaped around 0.5.

    We always take max(λ, 1−λ) so that x₁ always contributes at least 50% of
    the mix.  This is a common convention (followed in the original Mixup paper)
    that avoids redundancy: without it, (λ=0.3, perm) and (λ=0.7, perm⁻¹)
    would produce identical mixed examples, wasting sampling diversity.

    Parameters
    ----------
    alpha : float
        Shape parameter for the symmetric Beta distribution.

    Returns
    -------
    lam : float
        Mixing coefficient in [0.5, 1.0].
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    # Ensure x1 is always the "dominant" source (avoids redundant symmetric pairs)
    return max(lam, 1.0 - lam)


def apply_mixup(
    x: Tensor,
    y_onehot: Tensor,
    alpha: float = 0.4,
) -> tuple[Tensor, Tensor]:
    """Apply Vanilla Mixup to a batch of spectrograms.

    Pairs each example in the batch with a randomly permuted partner from
    the same batch, then blends both input and label with coefficient λ.

    Parameters
    ----------
    x : Tensor, shape (B, 1, H, W)
        Batch of spectrograms.  The channel dimension (C=1) is preserved.
    y_onehot : Tensor, shape (B, num_classes)
        One-hot (or already soft) label vectors.
    alpha : float
        Beta distribution parameter.  Default 0.4 per project specification.

    Returns
    -------
    x_mixed : Tensor, shape (B, 1, H, W)
        Mixed spectrograms.
    y_mixed : Tensor, shape (B, num_classes)
        Soft mixed labels: λ·y₁ + (1−λ)·y₂.

    Mathematical note
    -----------------
    The mixing operation is a simple weighted average in pixel space:
        x_mixed[b] = λ · x[b] + (1−λ) · x[perm[b]]
    Because the mel-spectrogram values are log-compressed (dB), this blending
    mixes log-energies linearly — not physical energies.  In audio terms this
    corresponds to mixing the log-spectrograms, which is *not* the same as
    mixing the waveforms and then taking the log-spectrogram.  However, this
    distinction applies equally to all blending-based augmentation methods
    (including CutMix and Puzzle Mix), so it does not affect the relative
    comparison between conditions.
    """
    batch_size = x.size(0)
    device = x.device

    lam = sample_lambda(alpha)

    # Random permutation of indices within the batch — each example is
    # paired with a different example (permutation, not replacement)
    perm = torch.randperm(batch_size, device=device)

    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_mixed = lam * y_onehot + (1.0 - lam) * y_onehot[perm]

    return x_mixed, y_mixed
