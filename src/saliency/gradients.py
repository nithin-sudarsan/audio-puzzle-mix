"""Vanilla gradient saliency computation.

Vanilla gradient saliency (Simonyan et al., 2013) defines the importance of
each input pixel as the magnitude of the gradient of the predicted class score
with respect to that pixel:

    S(x) = |∂f_{ŷ}(x) / ∂x|

where f_{ŷ}(x) is the logit (pre-softmax score) for the predicted class ŷ,
and the absolute value is taken elementwise.

Why vanilla gradients (and not Integrated Gradients, GradCAM, etc.)
--------------------------------------------------------------------
This project uses vanilla gradients for two reasons:

  1. Internal consistency: vanilla gradients are the signal Puzzle Mix uses
     internally (as ‖∇_x ℓ‖₂) to guide the graph-cut mask.  Evaluating
     Puzzle Mix with the same signal it was trained on creates a closed-loop
     test — we ask whether the signal it optimises becomes more perceptually
     faithful as a result of that optimisation.

  2. Transparency: vanilla gradients are implemented in ~5 lines of PyTorch
     autograd — no external libraries, no approximations, no reference baselines
     needed for comparison.  Every step is legible.

Using a single saliency method is a limitation (noted in Discussion): it is
possible that Puzzle Mix improves faithfulness on SmoothGrad or Integrated
Gradients but not on vanilla gradients, or vice versa.

Evaluation vs. training saliency
----------------------------------
In the training loop, saliency is computed as ‖∇_x ℓ‖₂ (L2 norm, all classes,
loss gradient) to give Puzzle Mix a global view of which input regions matter
for the classification.

At evaluation time, saliency is computed as |∂f_{ŷ}/∂x| (absolute elementwise
gradient of the predicted-class logit) to give a per-pixel importance map that
can be compared against the 2D class-conditional profile.

This file provides both variants:
  - `compute_eval_saliency`: post-training evaluation (per predicted class)
  - `compute_training_saliency`: training-time signal for Puzzle Mix

Interface
---------
Both functions accept a batch of spectrograms and a model, and return saliency
maps.  The model must be in eval mode for the eval variant (to freeze BN
statistics).  The model's gradient computation is temporarily re-enabled for
the input tensor only.
"""

import torch
import torch.nn as nn
from torch import Tensor


def compute_eval_saliency(
    model: nn.Module,
    x: Tensor,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Compute vanilla gradient saliency maps for evaluation.

    For each example in the batch, computes:

        S(x) = |∂f_{ŷ}(x) / ∂x|

    where ŷ = argmax f(x) is the model's predicted class and f_{ŷ} is the
    pre-softmax logit for that class.

    The model must be in eval() mode when this function is called.  This
    ensures BatchNorm uses its running statistics rather than batch statistics,
    which would otherwise make the gradient depend on other examples in the
    batch.

    The returned saliency maps are normalised to [0, 1] per example by dividing
    by the maximum value.  This makes maps comparable across examples and
    conditions without distorting the spatial distribution of importance.

    Parameters
    ----------
    model : nn.Module
        Trained model.  Must be in eval() mode.  Must output logits (no softmax).
    x : Tensor, shape (B, 1, H, W)
        Batch of spectrograms.  Does not need requires_grad=True; we set it here.
    device : str
        Computation device.

    Returns
    -------
    saliency : Tensor, shape (B, H, W)
        Normalised gradient saliency maps.  Values in [0, 1].  The channel
        dimension is squeezed out (single-channel input → single map per example).
    predicted_classes : Tensor, shape (B,)
        Predicted class indices (argmax of logits).  Returned so the caller can
        filter to correctly-classified examples.
    """
    # Move input to device and enable gradient tracking for this tensor only.
    # We do NOT call model.train() — eval mode must be preserved.
    x_input = x.to(device).detach().requires_grad_(True)

    # Forward pass: compute class logits
    # model is in eval() — BN and dropout use inference behaviour
    logits = model(x_input)   # (B, num_classes)

    # Predicted class for each example
    predicted_classes = logits.argmax(dim=1)   # (B,)

    # Gather the logit for each example's predicted class.
    # logits[b, predicted_classes[b]] for each b.
    # gather: select column predicted_classes[b] from row b → (B, 1)
    predicted_logits = logits.gather(1, predicted_classes.unsqueeze(1))   # (B, 1)

    # Backpropagate the *sum* of predicted logits.
    # Summing over the batch is equivalent to independent per-example gradients
    # when the model has no inter-example dependencies (which eval mode ensures).
    # The gradient ∂(Σ_b f_{ŷ_b}(x_b)) / ∂x_b = ∂f_{ŷ_b}(x_b) / ∂x_b for each b.
    predicted_logits.sum().backward()

    # x_input.grad shape: (B, 1, H, W) — same as x_input
    # Take absolute value elementwise: importance = magnitude of gradient
    grad = x_input.grad.detach().abs()   # (B, 1, H, W)

    # Squeeze the channel dimension (C=1 for single-channel spectrograms)
    grad = grad.squeeze(1)   # (B, H, W)

    # Normalise each map to [0, 1] by its own maximum.
    # Per-sample normalisation: the denominator is the max over all (H, W) positions.
    # Adding 1e-8 prevents division by zero for zero-gradient maps (e.g. padding regions).
    max_vals = grad.reshape(grad.shape[0], -1).max(dim=1).values   # (B,)
    max_vals = max_vals.clamp(min=1e-8)
    saliency = grad / max_vals.reshape(-1, 1, 1)   # (B, H, W), values in [0, 1]

    return saliency, predicted_classes


def compute_training_saliency(
    model: nn.Module,
    x: Tensor,
    y_onehot: Tensor,
    device: str,
) -> Tensor:
    """Compute saliency maps for use as the Puzzle Mix training signal.

    Computes the L2 norm of the input gradient with respect to the cross-entropy
    loss (summed over classes), as used in the original Puzzle Mix paper:

        s(x) = ‖∂ℓ(x, y) / ∂x‖₂

    where ℓ is the cross-entropy loss between model logits and the true
    one-hot labels, and ‖·‖₂ denotes the L2 norm over the channel dimension
    (trivially equal to the absolute value for single-channel inputs).

    This differs from the evaluation saliency in two ways:
      1. It uses the *loss* gradient (∂ℓ/∂x) rather than the logit gradient
         (∂f_{ŷ}/∂x).  The loss gradient reflects sensitivity to the correct
         class rather than the predicted class.
      2. It is computed in model.eval() mode — call the model in eval() before
         calling this function to ensure BN statistics are frozen.  The caller
         is responsible for restoring train() mode afterward.

    The returned maps are normalised to [0, 1] per example (same as eval saliency)
    so that Puzzle Mix can treat them as spatial importance distributions.

    Parameters
    ----------
    model : nn.Module
        Model to compute gradients through.  Should be in eval() mode
        (frozen BN) to prevent saliency from depending on other examples.
    x : Tensor, shape (B, 1, H, W)
        Batch of spectrograms.
    y_onehot : Tensor, shape (B, num_classes)
        One-hot (or soft) label vectors for the current batch.
    device : str
        Computation device.

    Returns
    -------
    saliency : Tensor, shape (B, H, W)
        Normalised gradient saliency maps.  Values in [0, 1].
        Detached from the computation graph (no gradient tracking in the
        returned tensor — the caller must not backpropagate through it).
    """
    x_input = x.to(device).detach().requires_grad_(True)
    y_target = y_onehot.to(device)

    # Forward pass
    logits = model(x_input)   # (B, num_classes)

    # Cross-entropy loss: -Σ_c y_c · log softmax(logits)_c, summed over batch
    # Using log_softmax + nll is numerically stable
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    loss = -(y_target * log_probs).sum()   # scalar

    # Backpropagate to get ∂ℓ/∂x
    loss.backward()

    # L2 norm over channel dimension (C=1 → same as absolute value)
    # grad: (B, 1, H, W) → squeeze to (B, H, W)
    grad = x_input.grad.detach().abs().squeeze(1)   # (B, H, W)

    # Normalise per example to [0, 1]
    max_vals = grad.reshape(grad.shape[0], -1).max(dim=1).values.clamp(min=1e-8)
    saliency = grad / max_vals.reshape(-1, 1, 1)

    return saliency
