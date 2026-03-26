"""CutMix augmentation for mel-spectrogram training.

CutMix (Yun et al., 2019) constructs a mixed training example by replacing a
rectangular region of one spectrogram with the corresponding region from
another, and mixing labels proportionally to the replaced area:

    x_mixed = M ⊙ x₁ + (1 − M) ⊙ x₂
    y_mixed = λ_actual · y₁ + (1 − λ_actual) · y₂

where M is a binary mask with a single rectangular hole, and λ_actual is the
fraction of x₁ that remains after the cut (1 − patch_area / total_area).

Unlike Vanilla Mixup, CutMix preserves the spatial structure of both source
examples within their respective regions.  The pasted patch retains the exact
spectral content of its source at the original time-frequency coordinates.

Why this baseline is informative
----------------------------------
CutMix preserves local spectral structure (each cell belongs entirely to one
class) but the patch location is random — it has no knowledge of which
frequency bands are class-discriminative.  From an XAI perspective, this
creates a weaker inductive bias than Puzzle Mix: the model sees intact spectral
features, but these features are mixed in spatially random ways that may cut
across class-defining frequency bands.  We expect CutMix to produce higher FBC
scores than Vanilla Mixup (intact local structure is preserved) but lower than
Puzzle Mix (the cut is not guided by saliency).  This expectation corresponds
to hypothesis H3 in the project proposal.

Patch size and label mixing
-----------------------------
λ ~ Beta(α, α) determines the expected fraction of the spectrogram that is
*kept* from x₁.  The patch (from x₂) occupies approximately (1 − λ) of the
total area.  Patch dimensions:

    patch_height = √(1 − λ) × H
    patch_width  = √(1 − λ) × W

Factoring √(1-λ) evenly across both dimensions means the patch is proportional
to the spectrogram's aspect ratio (important here since our spectrograms are
non-square at 128 × 500).  The patch centre is uniformly sampled; coordinates
are clipped to valid bounds.  The actual λ (after clipping) is recomputed from
the true patch area and used for label mixing, ensuring label proportions are
always consistent with the actual mix.

Interface contract
-------------------
Matches the contract described in mixup.py:
    x_mixed, y_mixed = apply_cutmix(x, y_onehot, alpha)
"""

import numpy as np
import torch
from torch import Tensor


def _sample_patch(
    height: int,
    width: int,
    lam: float,
) -> tuple[int, int, int, int]:
    """Sample a random rectangular patch for CutMix.

    The patch area is (1 − λ) × H × W.  The aspect ratio of the patch matches
    the aspect ratio of the spectrogram (H : W), so the patch is:
        patch_height = √(1 − λ) × H
        patch_width  = √(1 − λ) × W

    This keeps the patch proportional to the spectrogram dimensions.  For our
    non-square (128 × 500) spectrograms, a purely square patch would
    cover very different fractions of the frequency and time axes.

    The centre (cx, cy) is sampled uniformly over the full spectrogram, and
    the patch edges are clipped to [0, H] × [0, W] to stay within bounds.

    Parameters
    ----------
    height, width : int
        Spectrogram spatial dimensions.
    lam : float
        CutMix mixing coefficient (fraction of x₁ remaining after the cut).

    Returns
    -------
    (x1, x2, y1, y2) : int
        Pixel indices of the patch: rows [x1, x2), cols [y1, y2).
    """
    # Scale patch dimensions proportionally to the spectrogram shape.
    # √(1−λ) is the linear scaling factor: if we scale both H and W by
    # √(1−λ), the area scales by (1−λ) as required.
    scale = np.sqrt(1.0 - lam)
    patch_h = scale * height
    patch_w = scale * width

    # Random centre, uniformly distributed over the full spectrogram
    cx = np.random.uniform(0, height)
    cy = np.random.uniform(0, width)

    # Patch corners, clipped to valid bounds
    x1 = int(np.clip(cx - patch_h / 2, 0, height))
    x2 = int(np.clip(cx + patch_h / 2, 0, height))
    y1 = int(np.clip(cy - patch_w / 2, 0, width))
    y2 = int(np.clip(cy + patch_w / 2, 0, width))

    return x1, x2, y1, y2


def apply_cutmix(
    x: Tensor,
    y_onehot: Tensor,
    alpha: float = 0.4,
) -> tuple[Tensor, Tensor]:
    """Apply CutMix to a batch of spectrograms.

    A single patch location is sampled and applied uniformly across the entire
    batch.  This is equivalent to the original CutMix implementation and is
    more efficient than sampling a separate patch per example.

    Parameters
    ----------
    x : Tensor, shape (B, 1, H, W)
        Batch of spectrograms.
    y_onehot : Tensor, shape (B, num_classes)
        One-hot (or already soft) label vectors.
    alpha : float
        Beta distribution parameter.  Default 0.4 per project specification.

    Returns
    -------
    x_mixed : Tensor, shape (B, 1, H, W)
        Spectrograms with a rectangular patch replaced from the permuted partner.
    y_mixed : Tensor, shape (B, num_classes)
        Soft labels mixed proportionally to actual patch area.

    Implementation note
    --------------------
    x is modified in-place for the patch region to avoid allocating a full copy
    of the batch.  The input tensor is cloned before modification so the
    original batch is not changed.
    """
    batch_size, C, H, W = x.shape
    device = x.device

    # Sample λ from Beta(α, α); take max(λ, 1−λ) as in Vanilla Mixup
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    lam = max(lam, 1.0 - lam)

    # Sample the patch coordinates
    x1, x2, y1, y2 = _sample_patch(H, W, lam)

    # Random permutation to pair each example with a partner from the batch
    perm = torch.randperm(batch_size, device=device)

    # Clone so we don't modify the original batch tensor
    x_mixed = x.clone()

    # Replace the patch in each example with the corresponding patch from its
    # permuted partner.  The rest of x_mixed remains x unchanged.
    x_mixed[:, :, x1:x2, y1:y2] = x[perm, :, x1:x2, y1:y2]

    # Recompute the actual mixing coefficient from the true patch area.
    # Because patch edges are clipped, the actual patch area may be smaller
    # than (1−λ)×H×W.  Using the true area ensures label mixing is consistent
    # with what is actually in the image.
    patch_area = (x2 - x1) * (y2 - y1)
    lam_actual = 1.0 - patch_area / (H * W)

    y_mixed = lam_actual * y_onehot + (1.0 - lam_actual) * y_onehot[perm]

    return x_mixed, y_mixed
