"""ResNet-18 adapted for single-channel mel-spectrogram classification.

This module provides a ResNet-18 backbone modified to accept single-channel
input (mel-spectrograms) and produce 50-class output (ESC-50).

Two modifications are made to the standard ImageNet ResNet-18:

  1. First convolutional layer: in_channels changed from 3 → 1.
     The pretrained 3-channel weights are adapted by summing across the
     input-channel dimension (see _adapt_first_conv for full rationale).

  2. Final fully-connected layer: out_features changed from 1000 → 50.
     This layer is randomly re-initialised (pretrained ImageNet class weights
     are meaningless for ESC-50 classes).

All other layers and their pretrained weights are preserved exactly.

XAI note
--------
Starting from ImageNet-pretrained weights (rather than random initialisation)
matters for this project beyond classification accuracy.  During Puzzle Mix
training, the augmentation mask is determined by the *current model's* saliency
maps.  In early epochs, a randomly-initialised model produces essentially
random saliency maps, which means early Puzzle Mix masks are no better than
random CutMix masks.  A pretrained initialisation gives the model meaningful
feature detectors from epoch 1, so the Puzzle Mix feedback loop starts from a
more informative saliency signal.  This applies equally to all conditions
(the pretrained weights are shared), so it does not bias the comparison — but
it may reduce the number of epochs needed for the feedback loop to stabilise.

Spatial dimensions for (1, 128, 500) input
-------------------------------------------
  conv1 (7×7, s=2, p=3) : (64,  64, 250)
  maxpool (3×3, s=2, p=1): (64,  32, 125)
  layer2 (s=2)           : (128, 16,  63)
  layer3 (s=2)           : (256,  8,  32)
  layer4 (s=2)           : (512,  4,  16)
  AdaptiveAvgPool(1,1)   : (512,  1,   1)
  fc                     : (50)

The non-square feature maps (e.g. 4×16 before the final pool) are handled
transparently by AdaptiveAvgPool2d((1,1)), which averages over any spatial size.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def _adapt_first_conv(pretrained_conv: nn.Conv2d) -> nn.Conv2d:
    """Return a new Conv2d that accepts 1-channel input, with adapted weights.

    The original first conv has weight shape (64, 3, 7, 7): 64 filters, each
    operating over 3 input channels with a 7×7 kernel.

    To convert to 1-channel input while preserving as much pretrained
    information as possible, we **sum** the weights across the 3 input channels:

        new_weight[f, 0, h, w] = Σ_c  old_weight[f, c, h, w]   for c in {0,1,2}

    Why sum rather than average?
    The original filter computes:  activation = w₀·x₀ + w₁·x₁ + w₂·x₂
    For a single-channel input x:  activation = (w₀ + w₁ + w₂) · x
    Summing gives the same total sensitivity as the original 3-channel filter
    would have for a uniform-intensity input (x₀ = x₁ = x₂ = x).
    Averaging instead would scale the weights by 1/3, producing activations
    approximately 3× smaller than the pretrained BN/ReLU thresholds were
    calibrated for — potentially distorting early feature detection.

    Why does this matter for saliency?
    The saliency map in Puzzle Mix is computed as ‖∇ₓ ℓ‖₂ at training time.
    If the first conv weights are poorly scaled, gradients flowing back through
    it will be poorly scaled too, producing noisy Puzzle Mix masks in early
    epochs.  Starting from well-scaled weights gives the feedback loop a better
    initial signal.

    Parameters
    ----------
    pretrained_conv : nn.Conv2d
        The original 3-channel first conv from the pretrained ResNet-18.

    Returns
    -------
    new_conv : nn.Conv2d
        An identical Conv2d with in_channels=1 and weights summed from the
        original 3-channel weights.  bias=False (matching the original).
    """
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=pretrained_conv.out_channels,   # 64
        kernel_size=pretrained_conv.kernel_size,     # (7, 7)
        stride=pretrained_conv.stride,               # (2, 2)
        padding=pretrained_conv.padding,             # (3, 3)
        bias=False,                                  # ResNet-18 has no bias in conv1
    )

    # pretrained weight shape: (64, 3, 7, 7)
    # sum over dim=1 (input channels) → shape: (64, 1, 7, 7)
    with torch.no_grad():
        new_conv.weight.copy_(
            pretrained_conv.weight.sum(dim=1, keepdim=True)
        )

    return new_conv


def build_model(
    num_classes: int = 50,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet-18 adapted for single-channel spectrogram classification.

    This is the single entry point for model construction used throughout the
    project.  All four training conditions (baseline, vanilla_mixup, cutmix,
    puzzle_mix) call this function with the same arguments, ensuring that
    differences in saliency quality are attributable to the augmentation method
    rather than to any architectural variation.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  50 for ESC-50.
    pretrained : bool
        If True, load ImageNet-pretrained weights and adapt them for 1-channel
        input.  If False, initialise all weights randomly (useful for ablations
        that isolate the contribution of pretraining).

    Returns
    -------
    model : nn.Module
        Modified ResNet-18.  Input: (B, 1, 128, 500) float32 tensor.
        Output: (B, num_classes) logit tensor (no softmax applied — the
        training loop applies softmax+BCE or cross-entropy externally).

    Notes on the FC layer initialisation
    -------------------------------------
    The new FC layer (512 → num_classes) is initialised with PyTorch's default
    kaiming_uniform_ for weights and uniform_ for bias.  This is appropriate
    because the 50 ESC-50 classes have no correspondence to ImageNet classes,
    so the pretrained FC weights are discarded entirely.
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    # --- Modification 1: adapt first conv for 1-channel input ---
    if pretrained:
        model.conv1 = _adapt_first_conv(model.conv1)
    else:
        # Random init for 1-channel conv, keeping same spatial configuration
        model.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )

    # --- Modification 2: replace final FC with num_classes-class head ---
    # model.fc is Linear(512, 1000) in the standard ResNet-18.
    # in_features=512 comes from the 4th residual block's output channel count.
    in_features = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)
    # PyTorch's default initialisation (kaiming_uniform_) is left in place —
    # no special init is needed for a standard linear classification head.

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters in the model.

    Useful for reporting in the Experiments section of the paper and for
    sanity-checking that the modifications did not unexpectedly change the
    parameter count.

    Returns a dict with keys 'trainable', 'frozen', 'total'.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable,
        "frozen":    frozen,
        "total":     trainable + frozen,
    }
