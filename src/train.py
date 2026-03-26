"""Main training loop for all four augmentation conditions.

This script trains a ResNet-18 on ESC-50 mel-spectrograms under one of four
augmentation conditions:

  baseline      — random crop + horizontal flip, no mixing
  vanilla_mixup — Vanilla Mixup (linear pixel blend, λ ~ Beta(0.4, 0.4))
  cutmix        — CutMix (random rectangular patch swap)
  puzzle_mix    — Puzzle Mix (graph-cut mask + optional transport)

The training loop is shared across all conditions; the condition is selected
via the --condition command-line argument.  All other hyperparameters (LR,
schedule, optimizer, epochs, batch size) are fixed across conditions so that
any differences in outcome can be attributed to the augmentation alone.

Hyperparameters (fixed, per project specification)
---------------------------------------------------
  Optimiser:  SGD, momentum=0.9, weight_decay=1e-4
  LR:         0.1, cosine annealing to 0, over `epochs` epochs
  Epochs:     200 (default; can be reduced for debugging)
  Batch size: 64
  Seeds:      3 random seeds per condition (specified via --seed)
  Alpha:      0.4 (Beta distribution parameter for all mixing methods)

Reproducibility
---------------
All random state is seeded via set_seed() before the dataset and model are
created.  The seed applies to Python, NumPy, and PyTorch.  DataLoader workers
inherit their seed via worker_init_fn.  This ensures that for the same
(condition, seed) pair, the training trajectory is fully reproducible.

Checkpointing
-------------
After each epoch a checkpoint is saved to:
  results/<condition>/seed<seed>/epoch<epoch>.pt
containing the model state_dict, optimizer state_dict, epoch number, train
loss, and validation accuracy.  The best-validation-accuracy checkpoint is
also saved as best.pt in the same directory.

Progress logging
----------------
Training progress is printed to stdout as a one-line summary per epoch:
  Epoch 001 | train_loss=1.2345 | val_acc=0.3250 | lr=0.1000
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src/ to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from data.esc50 import get_dataloaders
from models.resnet import build_model
from augmentation.mixup import apply_mixup
from augmentation.cutmix import apply_cutmix
from augmentation.puzzle_mix import apply_puzzle_mix
from saliency.gradients import compute_training_saliency


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Seeds Python's random, NumPy, and PyTorch (CPU and CUDA/MPS).  Also sets
    the PYTHONHASHSEED environment variable so that dict ordering is
    deterministic even if the script is called from a subprocess.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Deterministic ops where available (may slow down MPS/CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker so data loading is reproducible.

    Called automatically by DataLoader for each worker process.  Without this,
    workers use the same NumPy seed, leading to identical random augmentation
    within each worker.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Select the best available compute device.

    Prefers CUDA > MPS (Apple Silicon) > CPU.  Returns the device string.
    """
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ---------------------------------------------------------------------------
# One-hot label encoding
# ---------------------------------------------------------------------------

def to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer class labels to one-hot vectors.

    All augmentation functions require one-hot labels (to support soft-label
    mixing).  This function converts a batch of integer labels to one-hot form.

    Parameters
    ----------
    labels : Tensor, shape (B,)
        Integer class indices in [0, num_classes).
    num_classes : int
        Total number of classes.

    Returns
    -------
    one_hot : Tensor, shape (B, num_classes)
        One-hot label matrix on the same device as `labels`.
    """
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return one_hot


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss supporting soft (non-integer) labels.

    Standard nn.CrossEntropyLoss requires integer labels.  All mixing methods
    produce soft labels (convex combinations of one-hot vectors), so we
    compute the loss manually:

        ℓ = −Σ_c y_c · log softmax(logit_c)

    This is the correct generalisation of cross-entropy to soft targets.

    Parameters
    ----------
    logits : Tensor, shape (B, C)
        Pre-softmax model outputs.
    soft_labels : Tensor, shape (B, C)
        Soft label vectors (should sum to 1 per example, but not enforced).

    Returns
    -------
    loss : Tensor, scalar
        Mean loss over the batch.
    """
    log_probs = F.log_softmax(logits, dim=1)   # (B, C)
    loss = -(soft_labels * log_probs).sum(dim=1)   # (B,) per-example loss
    return loss.mean()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader, device: str) -> float:
    """Compute top-1 accuracy on a data split.

    Parameters
    ----------
    model : nn.Module
        Model in eval() mode.
    loader : DataLoader
        Validation or test data loader.
    device : str

    Returns
    -------
    accuracy : float
        Fraction of correctly classified examples.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    condition: str,
    seed: int,
    data_root: str,
    results_root: str,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.1,
    alpha: float = 0.4,
    num_classes: int = 50,
    puzzle_block_sizes: tuple[int, ...] = (16, 32),
    save_every: int = 10,
) -> None:
    """Train ResNet-18 under one augmentation condition with one random seed.

    Parameters
    ----------
    condition : str
        One of 'baseline', 'vanilla_mixup', 'cutmix', 'puzzle_mix'.
    seed : int
        Random seed for full reproducibility.
    data_root : str
        Path to the ESC-50 data root (passed to get_dataloaders).
    results_root : str
        Directory under which checkpoints are saved.
        Saves to: results_root/condition/seed<seed>/
    epochs : int
        Number of training epochs.  200 per specification.
    batch_size : int
        Training batch size.  64 per specification.
    lr : float
        Initial learning rate.  0.1 per specification.
    alpha : float
        Beta distribution shape parameter for Mixup / CutMix / Puzzle Mix.
        0.4 per specification.
    num_classes : int
        Number of output classes.  50 for ESC-50.
    puzzle_block_sizes : tuple of int
        Block sizes sampled uniformly for Puzzle Mix.  Kim et al. sample from
        {16, 32} for their experiments; we follow the same convention.
    save_every : int
        Save a checkpoint every this many epochs (in addition to best.pt).

    Notes
    -----
    For Puzzle Mix, saliency is computed before each forward pass by switching
    the model to eval() mode and backpropagating the cross-entropy loss.  The
    model is then switched back to train() mode for the actual update step.
    This adds roughly 2× the per-batch compute cost for the Puzzle Mix condition.
    """
    assert condition in ('baseline', 'vanilla_mixup', 'cutmix', 'puzzle_mix'), (
        f"Unknown condition: {condition}"
    )

    # --- Setup ---
    set_seed(seed)
    device = get_device()
    print(f"[train] condition={condition}  seed={seed}  device={device}  epochs={epochs}")

    # Output directory
    out_dir = Path(results_root) / condition / f'seed{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,           # 0 workers keeps MPS-friendly; increase if needed
        seed=seed,
    )

    # --- Model ---
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    # --- Optimiser + schedule ---
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    # --- Training loop ---
    best_val_acc = 0.0

    # Log hyperparameters to file for reproducibility
    hparam_path = out_dir / 'hparams.txt'
    with open(hparam_path, 'w') as f:
        f.write(f"condition={condition}\n")
        f.write(f"seed={seed}\n")
        f.write(f"device={device}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"lr={lr}\n")
        f.write(f"alpha={alpha}\n")
        f.write(f"num_classes={num_classes}\n")
        f.write(f"optimizer=SGD momentum=0.9 weight_decay=1e-4 nesterov=True\n")
        f.write(f"scheduler=CosineAnnealingLR T_max={epochs} eta_min=0.0\n")
        if condition == 'puzzle_mix':
            f.write(f"puzzle_block_sizes={puzzle_block_sizes}\n")
            f.write("puzzle_beta=1.2  puzzle_gamma=0.5  puzzle_eta=0.2  puzzle_t_eps=0.8  puzzle_n_labels=3\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            y_onehot = to_onehot(labels, num_classes)

            # --- Apply augmentation ---
            if condition == 'baseline':
                # No mixing augmentation; use the batch as-is.
                # Standard spatial augmentations (crop, flip) are not applied
                # here because ESC-50 spectrograms are not padded for crop —
                # see note in CLAUDE.md: the augmentation is purely the mixing
                # strategy.  The baseline condition isolates the effect of mixing.
                x_aug = x
                y_aug = y_onehot

            elif condition == 'vanilla_mixup':
                x_aug, y_aug = apply_mixup(x, y_onehot, alpha=alpha)

            elif condition == 'cutmix':
                x_aug, y_aug = apply_cutmix(x, y_onehot, alpha=alpha)

            elif condition == 'puzzle_mix':
                # Step 1: compute saliency in eval() mode (frozen BN statistics)
                # so the gradient signal is not contaminated by batch statistics
                # of the current training batch.
                model.eval()
                with torch.enable_grad():
                    saliency = compute_training_saliency(model, x, y_onehot, device)
                # saliency shape: (B, H, W), normalised to [0, 1] per example

                # Step 2: apply Puzzle Mix (graph cut + transport)
                model.train()
                block_size = int(np.random.choice(puzzle_block_sizes))
                x_aug, y_aug = apply_puzzle_mix(
                    x, y_onehot, saliency,
                    alpha=alpha,
                    beta=1.2,
                    gamma=0.5,
                    eta=0.2,
                    neigh_size=4,
                    n_labels=3,
                    use_transport=True,
                    t_eps=0.8,
                    block_size=block_size,
                    device=device,
                )

            # --- Forward + backward ---
            # Ensure model is in train mode before computing the actual gradient update
            model.train()
            optimizer.zero_grad()
            logits = model(x_aug)
            loss = soft_cross_entropy(logits, y_aug)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- Validate ---
        val_acc = evaluate(model, val_loader, device)

        # --- Log ---
        elapsed = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={epoch_loss/n_batches:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f} | "
            f"time={elapsed:.1f}s"
        )

        # --- Checkpoint ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_loss / n_batches,
            'val_acc': val_acc,
            'condition': condition,
            'seed': seed,
        }

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, out_dir / 'best.pt')

        # Periodic checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            torch.save(checkpoint, out_dir / f'epoch{epoch:03d}.pt')

    # --- Final test evaluation ---
    # Load the best checkpoint before evaluating on the test set.
    best_ckpt = torch.load(out_dir / 'best.pt', map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    test_acc = evaluate(model, test_loader, device)
    print(f"\nTest accuracy (best val checkpoint, epoch {best_ckpt['epoch']}): {test_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save final summary
    summary = {
        'condition': condition,
        'seed': seed,
        'best_val_acc': best_val_acc,
        'best_epoch': best_ckpt['epoch'],
        'test_acc': test_acc,
    }
    torch.save(summary, out_dir / 'summary.pt')

    # Also write as plain text for quick inspection
    with open(out_dir / 'summary.txt', 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for the training script.

    All hyperparameters are exposed as arguments with sensible defaults so
    that the script can be run in a single line for each (condition, seed)
    combination.  The key arguments are:

      --condition : which augmentation to use
      --seed      : random seed (for reproducibility across 3 runs)
      --data-root : path to ESC-50 data (default: ./data/ESC-50)
      --epochs    : number of training epochs (default: 200)
    """
    parser = argparse.ArgumentParser(
        description='Train ResNet-18 on ESC-50 under one augmentation condition.'
    )
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        choices=['baseline', 'vanilla_mixup', 'cutmix', 'puzzle_mix'],
        help='Augmentation condition to use for training.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility. Run 3 seeds (e.g. 42, 43, 44).',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/ESC-50',
        help='Root directory for ESC-50 dataset (will download if absent).',
    )
    parser.add_argument(
        '--results-root',
        type=str,
        default='results',
        help='Directory for checkpoints and summaries.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size (default: 64).',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Initial learning rate (default: 0.1).',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='Beta distribution parameter for mixing (default: 0.4).',
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save a checkpoint every N epochs (default: 10).',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        condition=args.condition,
        seed=args.seed,
        data_root=args.data_root,
        results_root=args.results_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        save_every=args.save_every,
    )
