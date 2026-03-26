# Saliency-Guided Augmentation for Explainable Audio Classification

MPhil Advanced Computer Science — Explainable Artificial Intelligence (L193)
Cambridge, Lent Term

---

## Overview

This project investigates whether training with **Puzzle Mix** — a saliency-aware data augmentation method — produces models whose explanations are more perceptually faithful. The domain is environmental sound classification using mel-spectrograms from the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50).

The central question: does Puzzle Mix's feedback loop, where saliency guides augmentation which in turn shapes the model, lead to saliency maps that better align with the characteristic spectral structure of each sound class?

---

## Methods

Four training conditions are compared, all using a ResNet-18 backbone pretrained on ImageNet:

| Condition | Augmentation |
|---|---|
| Baseline | No mixing augmentation |
| Vanilla Mixup | Linear pixel blend (λ ~ Beta(0.4, 0.4)) |
| CutMix | Random rectangular patch swap |
| Puzzle Mix | Saliency-guided mask + optimal transport |

**Input**: Log-mel spectrograms (128 × 500), computed with a 25ms window and 10ms hop.

---

## Evaluation

Saliency maps (vanilla gradients) are evaluated against class-conditional spectral profiles — averages of normalised mel-spectrogram energy per class, computed from training data only.

Three metrics:
- **Frequency Band Concentration (FBC)** — IoU between top-20% saliency and top-20% class profile
- **Spectral Pointing Game (SPG)** — proportion of examples where peak saliency falls in the class profile's top-20%
- **Saliency Sharpness** — Shannon entropy of the saliency map (lower = more concentrated)

---

## Results

| Condition | Test Acc | FBC | SPG | Sharpness (nats) |
|---|---|---|---|---|
| Baseline | 0.685 | 0.169 ± 0.107 | 0.331 | 10.339 ± 0.286 |
| Vanilla Mixup | 0.688 | 0.140 ± 0.050 | 0.325 | 10.485 ± 0.097 |
| CutMix | 0.690 | 0.186 ± 0.116 | 0.281 | 10.246 ± 0.306 |
| Puzzle Mix | 0.630 | 0.197 ± 0.116 | 0.336 | 10.318 ± 0.301 |

Puzzle Mix achieves the highest FBC and SPG, supporting a weak but real inductive bias toward perceptual faithfulness. Vanilla Mixup falls below the no-augmentation baseline on FBC, highlighting a disconnect between classification accuracy and explanation quality.

---

## Project Structure

```
project/
├── src/
│   ├── data/              # ESC-50 loading and mel-spectrogram conversion
│   ├── models/            # ResNet-18 adapted for single-channel input
│   ├── augmentation/      # Puzzle Mix, CutMix, Vanilla Mixup implementations
│   ├── saliency/          # Saliency computation and evaluation metrics
│   ├── train.py           # Training loop for all four conditions
│   └── evaluate.py        # Post-training saliency evaluation pipeline
├── experiments/           # Config files, one per condition
├── results/               # Metric CSVs and evaluation outputs
└── PuzzleMix/             # Reference implementation (Kim et al., ICML 2020)
```

---

## Dependencies

- Python 3.8+
- PyTorch + torchaudio
- pygco / gco-wrapper (graph cuts)
- scipy, numpy, matplotlib

---

## References

- Kim et al. (2020). *Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup*. ICML.
- Piczak (2015). *ESC: Dataset for Environmental Sound Classification*. ACM MM.
- Zhang et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
- Yun et al. (2019). *CutMix: Training Strategy that Makes Use of Sample Mixing*. ICCV.
