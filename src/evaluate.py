"""Post-training saliency evaluation pipeline.

This script loads a trained model checkpoint, computes vanilla gradient
saliency maps for all correctly-classified test examples, and evaluates
three perceptual faithfulness metrics:

  FBC (Frequency Band Concentration):
    IoU between the top-20% saliency mask and the top-20% class profile.

  SPG (Spectral Pointing Game):
    Hit rate: does the peak saliency cell fall in the top-20% of the class profile?

  Saliency Sharpness:
    Shannon entropy of the normalised saliency map.  Lower = more concentrated.

Results are saved as CSV files for downstream analysis and figure generation.

Usage
-----
Evaluate a single (condition, seed) pair:
  python src/evaluate.py \\
      --condition puzzle_mix \\
      --seed 42 \\
      --data-root data/ESC-50 \\
      --results-root results

Or evaluate all conditions and seeds together (calls the function directly):
  python src/evaluate.py --all --data-root data/ESC-50 --results-root results

Output files
------------
For each (condition, seed), saves to results/<condition>/seed<seed>/:
  saliency_results.csv   — per-example: class, fbc, spg_hit, sharpness
  metrics_summary.txt    — per-condition summary (mean ± std)

For the --all flag, also saves:
  results/all_conditions_summary.csv — all conditions × seeds in one table

Design note: only correctly-classified examples
-----------------------------------------------
We restrict evaluation to correctly-classified test examples.  On a
misclassified example, the predicted class (used for the gradient target)
is wrong — comparing saliency against the true-class profile would measure
alignment between two mismatched things.  More importantly, interpretability
research conventionally focuses on cases where the model is "right" — we want
to understand what the model learned, not its failure modes.

Design note: class profiles are built from training data only
--------------------------------------------------------------
Class profiles are averaged over folds 1–3 (training set only) to ensure
they are independent of the test examples.  Using test data to build profiles
would create a circularity: the test examples themselves could shift the
profile to match their own statistics.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add src/ to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from data.esc50 import get_dataloaders, compute_class_profiles
from models.resnet import build_model
from saliency.gradients import compute_eval_saliency
from saliency.metrics import evaluate_saliency


# ---------------------------------------------------------------------------
# Evaluate one (condition, seed) pair
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition: str,
    seed: int,
    data_root: str,
    results_root: str,
    num_classes: int = 50,
    top_frac: float = 0.2,
    batch_size: int = 32,
) -> dict:
    """Run the full saliency evaluation for one (condition, seed) checkpoint.

    Loads the best-checkpoint for this (condition, seed), runs inference on
    the full test set, collects saliency maps for correctly-classified examples,
    computes FBC / SPG / Sharpness, saves per-example CSV, and returns
    a summary dict.

    Parameters
    ----------
    condition : str
        One of 'baseline', 'vanilla_mixup', 'cutmix', 'puzzle_mix'.
    seed : int
        Random seed index.
    data_root : str
        Root directory for ESC-50 dataset.
    results_root : str
        Directory containing checkpoints (results/<condition>/seed<seed>/best.pt).
    num_classes : int
        Number of classes (50 for ESC-50).
    top_frac : float
        Top-k fraction for FBC and SPG binary masks (default 0.2 = top 20%).
    batch_size : int
        Batch size for saliency computation.  Smaller than training batch to
        fit gradient computation in memory.

    Returns
    -------
    summary : dict
        Keys: condition, seed, n_correct, fbc_mean, fbc_std, spg_hit_rate,
              sharpness_mean, sharpness_std, best_epoch, test_acc
    """
    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if (hasattr(torch.backends, 'mps') and
                             torch.backends.mps.is_available())
              else 'cpu')

    ckpt_path = Path(results_root) / condition / f'seed{seed}' / 'best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train.py --condition {condition} --seed {seed} first."
        )

    # --- Load model ---
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_epoch = checkpoint.get('epoch', '?')
    test_acc_at_best = checkpoint.get('val_acc', float('nan'))
    print(f"[evaluate] {condition}/seed{seed}: loaded epoch {best_epoch}, "
          f"val_acc={test_acc_at_best:.4f}")

    # --- Data ---
    _, _, test_loader, _ = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
    )

    # --- Class profiles (training data only) ---
    # compute_class_profiles builds profiles from the training split internally
    print(f"[evaluate] Building class profiles from training data...")
    class_profiles = compute_class_profiles(data_root=data_root)   # (50, 128, 500)
    class_profiles = class_profiles.to(device)

    # --- Collect saliency maps for correctly-classified examples ---
    all_saliency = []
    all_true_classes = []
    n_total = 0
    n_correct = 0

    print(f"[evaluate] Computing saliency maps on test set...")
    for x, labels in test_loader:
        x, labels = x.to(device), labels.to(device)
        batch_size_actual = x.shape[0]
        n_total += batch_size_actual

        # compute_eval_saliency sets model.eval() internally — it does not
        # mutate the model's train/eval state, just uses it in eval mode.
        # Since we already called model.eval() above, this is consistent.
        saliency, predicted = compute_eval_saliency(model, x, device)
        # saliency: (B, H, W); predicted: (B,)

        # Filter to correctly-classified examples only
        correct_mask = (predicted == labels)
        n_correct += correct_mask.sum().item()

        for i in range(batch_size_actual):
            if correct_mask[i]:
                all_saliency.append(saliency[i].cpu())
                all_true_classes.append(labels[i].item())

    print(f"[evaluate] Correctly classified: {n_correct}/{n_total} "
          f"({100*n_correct/n_total:.1f}%)")

    if n_correct == 0:
        print(f"[evaluate] WARNING: no correctly classified examples — skipping metrics.")
        return {
            'condition': condition, 'seed': seed, 'n_correct': 0,
            'fbc_mean': float('nan'), 'fbc_std': float('nan'),
            'spg_hit_rate': float('nan'),
            'sharpness_mean': float('nan'), 'sharpness_std': float('nan'),
            'best_epoch': best_epoch, 'test_acc': float('nan'),
        }

    # --- Compute metrics ---
    print(f"[evaluate] Computing FBC, SPG, Sharpness on {n_correct} examples...")
    results = evaluate_saliency(
        saliency_maps=all_saliency,
        class_profiles=class_profiles.cpu(),
        true_classes=all_true_classes,
        top_frac=top_frac,
    )

    # --- Save per-example CSV ---
    out_dir = Path(results_root) / condition / f'seed{seed}'
    csv_path = out_dir / 'saliency_results.csv'

    fbc_list = results['fbc']['per_example']
    spg_list = results['spg']['per_example']
    sharp_list = results['sharpness']['per_example']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'fbc', 'spg_hit', 'sharpness'])
        for cls, fbc, spg_hit, sharp in zip(
            all_true_classes, fbc_list, spg_list, sharp_list
        ):
            writer.writerow([cls, f'{fbc:.6f}', int(spg_hit), f'{sharp:.6f}'])

    print(f"[evaluate] Saved per-example results to {csv_path}")

    # --- Summary ---
    summary = {
        'condition': condition,
        'seed': seed,
        'n_correct': n_correct,
        'fbc_mean': results['fbc']['mean'],
        'fbc_std': results['fbc']['std'],
        'spg_hit_rate': results['spg']['hit_rate'],
        'sharpness_mean': results['sharpness']['mean'],
        'sharpness_std': results['sharpness']['std'],
        'best_epoch': best_epoch,
        'test_acc': test_acc_at_best,
    }

    # Save summary as plain text
    summary_path = out_dir / 'metrics_summary.txt'
    with open(summary_path, 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")

    print(f"[evaluate] FBC={summary['fbc_mean']:.4f}±{summary['fbc_std']:.4f}  "
          f"SPG={summary['spg_hit_rate']:.4f}  "
          f"Sharpness={summary['sharpness_mean']:.4f}±{summary['sharpness_std']:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Evaluate all conditions and seeds
# ---------------------------------------------------------------------------

def evaluate_all(
    conditions: list[str],
    seeds: list[int],
    data_root: str,
    results_root: str,
) -> None:
    """Evaluate all (condition, seed) combinations and save a combined CSV.

    Iterates over all combinations; skips any whose checkpoint is not found
    (so partial results can still be aggregated).

    Parameters
    ----------
    conditions : list of str
        Condition names to evaluate.
    seeds : list of int
        Seeds to evaluate for each condition.
    data_root, results_root : str
        Paths as in evaluate_condition.
    """
    all_summaries = []
    for condition in conditions:
        for seed in seeds:
            try:
                summary = evaluate_condition(
                    condition=condition,
                    seed=seed,
                    data_root=data_root,
                    results_root=results_root,
                )
                all_summaries.append(summary)
            except FileNotFoundError as e:
                print(f"[evaluate_all] Skipping {condition}/seed{seed}: {e}")

    if not all_summaries:
        print("[evaluate_all] No results found. Run training first.")
        return

    # Save combined CSV
    out_path = Path(results_root) / 'all_conditions_summary.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\n[evaluate_all] Combined summary saved to {out_path}")

    # Print a quick summary table
    print("\n" + "=" * 80)
    print(f"{'Condition':<15} {'Seed':>4}  {'FBC':>8}  {'SPG':>8}  {'Sharpness':>10}")
    print("-" * 80)
    for s in all_summaries:
        print(
            f"{s['condition']:<15} {s['seed']:>4}  "
            f"{s['fbc_mean']:>8.4f}  "
            f"{s['spg_hit_rate']:>8.4f}  "
            f"{s['sharpness_mean']:>10.4f}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate saliency faithfulness metrics for trained models.'
    )
    parser.add_argument(
        '--condition',
        type=str,
        choices=['baseline', 'vanilla_mixup', 'cutmix', 'puzzle_mix'],
        help='Single condition to evaluate (use --all to evaluate all).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed index to evaluate (use --all to evaluate all seeds).',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Evaluate all conditions × seeds.',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 43, 44],
        help='Seeds to use with --all (default: 42 43 44).',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/ESC-50',
    )
    parser.add_argument(
        '--results-root',
        type=str,
        default='results',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    all_conditions = ['baseline', 'vanilla_mixup', 'cutmix', 'puzzle_mix']

    if args.all:
        evaluate_all(
            conditions=all_conditions,
            seeds=args.seeds,
            data_root=args.data_root,
            results_root=args.results_root,
        )
    elif args.condition and args.seed is not None:
        evaluate_condition(
            condition=args.condition,
            seed=args.seed,
            data_root=args.data_root,
            results_root=args.results_root,
        )
    else:
        print("Specify either --all or both --condition and --seed.")
        print("Example: python src/evaluate.py --condition puzzle_mix --seed 42")
