"""Saliency evaluation metrics: FBC, SPG, and Saliency Sharpness.

These three metrics operationalise the concept of *perceptual faithfulness* —
whether a model's saliency maps concentrate on the spectral features that are
independently knowable as class-defining.

All three metrics compare saliency maps against per-class spectral profiles
(computed in src/data/esc50.py::compute_class_profiles), which are constructed
from average mel-spectrogram energy across training clips of each class.  The
profiles are a data-driven proxy for perceptual salience: they capture which
time-frequency cells are consistently active for a given class, without
involving the model at all.

Metric overview
---------------
FBC (Frequency Band Concentration)
  IoU between the top-20% saliency mask and the top-20% class profile mask.
  Measures whether the model's top-saliency region overlaps with the class's
  characteristic spectral region.
  Range: [0, 1].  Higher is more perceptually faithful.

SPG (Spectral Pointing Game)
  Hit rate: does the single peak saliency cell fall in the top-20% of the
  class-conditional profile?
  Range: [0, 1].  Higher is more perceptually faithful.

Saliency Sharpness (secondary)
  Entropy of the normalised saliency map: H(S) = −Σ S(i) log S(i).
  Measures how concentrated/localised the saliency is, independently of
  whether it is in the right location.
  Range: [0, log(H*W)].  Lower is more concentrated.

Threshold choice (top-20%)
---------------------------
The top-20% threshold is applied independently to each saliency map and
to each class profile.  This means the number of "important" cells is fixed
at 20% × H × W = 12,800 cells for our (128 × 500) spectrograms.  Fixing the
threshold ensures IoU and hit-rate comparisons are not confounded by maps of
different sparsity.

Batch interface
---------------
All three metrics are designed to be computed over a list of (saliency, class,
is_correct) tuples aggregated from the test set, and return per-class and
overall statistics.  This matches the evaluation pipeline in src/evaluate.py.
"""

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Frequency Band Concentration (FBC)
# ---------------------------------------------------------------------------

def compute_fbc(
    saliency: Tensor,
    class_profile: Tensor,
    top_frac: float = 0.2,
) -> float:
    """Compute the Frequency Band Concentration (FBC) for one example.

    FBC is defined as the Intersection-over-Union (IoU) between two binary
    masks:
      - importance_mask: top-20% cells of the saliency map S(x)
      - reference_mask:  top-20% cells of the class-conditional profile P(c)

    IoU = |importance ∩ reference| / |importance ∪ reference|

    A high FBC means the model's most-salient cells spatially overlap with
    the cells that are consistently active for the ground-truth class.  A low
    FBC means the model attends to regions that are not class-discriminative
    according to the perceptual reference.

    Parameters
    ----------
    saliency : Tensor, shape (H, W)
        Normalised saliency map for a single example.  Values in [0, 1].
    class_profile : Tensor, shape (H, W)
        Class-conditional spectral profile for the example's true class.
        Shape must match saliency.
    top_frac : float
        Fraction of cells to include in each binary mask.  Default 0.2 (top 20%).

    Returns
    -------
    fbc : float
        IoU in [0, 1].  Returns 0.0 if neither mask has any active cells
        (degenerate case — should not occur in practice for top_frac > 0).
    """
    assert saliency.shape == class_profile.shape, (
        f"saliency shape {saliency.shape} != profile shape {class_profile.shape}"
    )
    n_cells = saliency.numel()
    k = max(1, int(top_frac * n_cells))   # number of cells in each mask

    # Flatten to 1D for easier top-k indexing
    sal_flat = saliency.reshape(-1)
    prof_flat = class_profile.reshape(-1)

    # Binary masks: True at the top-k positions
    # topk returns sorted values and indices; we only need the indices
    sal_idx = sal_flat.topk(k).indices
    prof_idx = prof_flat.topk(k).indices

    # Convert to boolean masks for set operations
    sal_mask = torch.zeros(n_cells, dtype=torch.bool, device=saliency.device)
    prof_mask = torch.zeros(n_cells, dtype=torch.bool, device=saliency.device)
    sal_mask[sal_idx] = True
    prof_mask[prof_idx] = True

    # IoU: intersection / union
    intersection = (sal_mask & prof_mask).sum().item()
    union = (sal_mask | prof_mask).sum().item()

    if union == 0:
        return 0.0
    return intersection / union


def compute_fbc_batch(
    saliency_maps: list[Tensor],
    class_profiles: Tensor,
    true_classes: list[int],
    top_frac: float = 0.2,
) -> dict:
    """Compute FBC for a list of test examples.

    Aggregates FBC scores per class and overall.

    Parameters
    ----------
    saliency_maps : list of Tensor, each (H, W)
        Saliency maps for each test example (should already be normalised to [0,1]).
    class_profiles : Tensor, shape (num_classes, H, W)
        Class-conditional profiles.  Indexed by class index.
    true_classes : list of int
        Ground-truth class index for each example.
    top_frac : float
        Top fraction for binary masks.

    Returns
    -------
    results : dict with keys:
        'per_example' : list of float — FBC for each example
        'per_class'   : dict {class_idx: mean FBC} — averaged within each class
        'mean'        : float — mean FBC over all examples
        'std'         : float — std FBC over all examples
    """
    per_example = []
    per_class_scores: dict[int, list[float]] = {}

    for sal, cls in zip(saliency_maps, true_classes):
        profile = class_profiles[cls]
        fbc = compute_fbc(sal, profile, top_frac=top_frac)
        per_example.append(fbc)
        per_class_scores.setdefault(cls, []).append(fbc)

    per_class_mean = {cls: float(np.mean(scores))
                      for cls, scores in per_class_scores.items()}
    all_scores = np.array(per_example)

    return {
        'per_example': per_example,
        'per_class': per_class_mean,
        'mean': float(all_scores.mean()),
        'std': float(all_scores.std()),
    }


# ---------------------------------------------------------------------------
# Spectral Pointing Game (SPG)
# ---------------------------------------------------------------------------

def compute_spg(
    saliency: Tensor,
    class_profile: Tensor,
    top_frac: float = 0.2,
) -> bool:
    """Compute the Spectral Pointing Game (SPG) hit for one example.

    SPG asks: does the single cell with the highest saliency value fall within
    the top-20% of the class-conditional profile?

    This is a stricter test than FBC: we require the *peak* saliency cell to
    be perceptually meaningful, not just that the 20% mass overlaps.  It is
    analogous to the Pointing Game metric used in weakly-supervised localisation
    (Zhang et al., 2018) but applied in the spectrogram domain.

    Parameters
    ----------
    saliency : Tensor, shape (H, W)
        Normalised saliency map for a single example.
    class_profile : Tensor, shape (H, W)
        Class-conditional spectral profile for the true class.
    top_frac : float
        Fraction of cells to consider as the reference region.  Default 0.2.

    Returns
    -------
    hit : bool
        True if the peak saliency cell is within the top-20% of the profile.
    """
    assert saliency.shape == class_profile.shape

    n_cells = saliency.numel()
    k = max(1, int(top_frac * n_cells))

    # Peak saliency cell index
    peak_idx = saliency.reshape(-1).argmax().item()

    # Top-k reference cells
    prof_flat = class_profile.reshape(-1)
    prof_top_idx = prof_flat.topk(k).indices

    # Hit: peak index appears in the reference top-k set
    return peak_idx in prof_top_idx.tolist()


def compute_spg_batch(
    saliency_maps: list[Tensor],
    class_profiles: Tensor,
    true_classes: list[int],
    top_frac: float = 0.2,
) -> dict:
    """Compute SPG hit rate for a list of test examples.

    Parameters
    ----------
    saliency_maps : list of Tensor, each (H, W)
    class_profiles : Tensor, shape (num_classes, H, W)
    true_classes : list of int
    top_frac : float

    Returns
    -------
    results : dict with keys:
        'per_example' : list of bool — hit indicator per example
        'per_class'   : dict {class_idx: hit_rate} — proportion of hits per class
        'hit_rate'    : float — overall proportion of hits
    """
    per_example = []
    per_class_hits: dict[int, list[bool]] = {}

    for sal, cls in zip(saliency_maps, true_classes):
        profile = class_profiles[cls]
        hit = compute_spg(sal, profile, top_frac=top_frac)
        per_example.append(hit)
        per_class_hits.setdefault(cls, []).append(hit)

    per_class_rate = {cls: float(np.mean(hits))
                      for cls, hits in per_class_hits.items()}

    return {
        'per_example': per_example,
        'per_class': per_class_rate,
        'hit_rate': float(np.mean(per_example)),
    }


# ---------------------------------------------------------------------------
# Saliency Sharpness (entropy)
# ---------------------------------------------------------------------------

def compute_sharpness(saliency: Tensor, eps: float = 1e-9) -> float:
    """Compute the saliency sharpness of one saliency map as normalised entropy.

    Sharpness is defined as the Shannon entropy of the saliency map, treated
    as a probability distribution over spatial positions:

        H(S) = −Σ_i p_i · log(p_i)

    where p_i = S_i / Σ_j S_j (normalised to sum to 1).

    Lower entropy means the saliency is more concentrated — the model assigns
    most of its importance weight to a small number of cells.  Higher entropy
    means the importance is diffuse — spread more evenly across the spectrogram.

    Note: this entropy is taken over the *spatial* distribution of saliency,
    not over class predictions.  It measures localisation quality, not
    prediction confidence.

    Why entropy (not e.g. sparsity ratio or peak fraction):
    Entropy accounts for the full distribution of saliency values, not just
    whether a fixed number of cells are above a threshold.  Two maps with
    identical top-20% IoU can have very different full-distribution sharpness.

    Parameters
    ----------
    saliency : Tensor, shape (H, W)
        Saliency map.  Need not be normalised to sum to 1; normalisation is
        done internally.
    eps : float
        Small constant added to saliency values before computing log to prevent
        log(0).  Saliency maps should not have exactly zero values after
        gradient computation, but this guard is included for safety.

    Returns
    -------
    entropy : float
        Shannon entropy in nats.  Range: [0, log(H × W)].
        0 = all mass on one cell (maximally sharp).
        log(H × W) = uniform distribution (maximally diffuse).
        For our (128 × 500) = 64,000-cell maps: max entropy ≈ 11.07 nats.
    """
    sal_flat = saliency.reshape(-1).float()
    sal_flat = sal_flat + eps                       # prevent log(0)
    prob = sal_flat / sal_flat.sum()                # normalise to probability distribution
    entropy = -(prob * prob.log()).sum().item()      # Shannon entropy in nats
    return entropy


def compute_sharpness_batch(saliency_maps: list[Tensor]) -> dict:
    """Compute saliency sharpness for a list of test examples.

    Parameters
    ----------
    saliency_maps : list of Tensor, each (H, W)
        Saliency maps for test examples (values need not sum to 1).

    Returns
    -------
    results : dict with keys:
        'per_example' : list of float — entropy per example
        'mean'        : float — mean entropy
        'std'         : float — std entropy
    """
    entropies = [compute_sharpness(sal) for sal in saliency_maps]
    arr = np.array(entropies)
    return {
        'per_example': entropies,
        'mean': float(arr.mean()),
        'std': float(arr.std()),
    }


# ---------------------------------------------------------------------------
# Combined evaluation helper
# ---------------------------------------------------------------------------

def evaluate_saliency(
    saliency_maps: list[Tensor],
    class_profiles: Tensor,
    true_classes: list[int],
    top_frac: float = 0.2,
) -> dict:
    """Compute all three metrics for a set of test examples.

    Convenience wrapper that calls compute_fbc_batch, compute_spg_batch, and
    compute_sharpness_batch in one call and returns a unified results dict.

    Parameters
    ----------
    saliency_maps : list of Tensor, each (H, W)
        Normalised saliency maps for test examples.
    class_profiles : Tensor, shape (num_classes, H, W)
        Class-conditional spectral profiles (from compute_class_profiles).
    true_classes : list of int
        Ground-truth class indices.
    top_frac : float
        Top-k fraction for FBC and SPG binary masks.

    Returns
    -------
    results : dict with keys 'fbc', 'spg', 'sharpness', each a sub-dict
        from their respective batch functions.
    """
    fbc_results = compute_fbc_batch(saliency_maps, class_profiles, true_classes, top_frac)
    spg_results = compute_spg_batch(saliency_maps, class_profiles, true_classes, top_frac)
    sharpness_results = compute_sharpness_batch(saliency_maps)
    return {
        'fbc': fbc_results,
        'spg': spg_results,
        'sharpness': sharpness_results,
    }
