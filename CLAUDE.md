# Puzzle Mix — XAI MPhil Project
**Cambridge MPhil, Explainable AI module | Lent Term**

---

## What This Project Is

This is an MPhil research project investigating whether training with a saliency-aware augmentation method (Puzzle Mix) produces models whose saliency maps are more *perceptually faithful* — that is, more concentrated on features that are independently knowable as class-defining.

The domain is **environmental sound classification using mel-spectrograms** (ESC-50 dataset). Audio spectrograms are treated as single-channel images; Puzzle Mix applies directly with no architectural changes.

The central research question:
> Does Puzzle Mix's training feedback loop — where saliency guides augmentation, which in turn shapes the model — produce models that are more explainable, as measured by alignment between their saliency maps and class-conditional spectral profiles?

This is an **XAI project first, a classification project second**. Accuracy numbers matter but they are not the primary contribution. The contribution is interpretive: using audio as a controlled testbed for understanding whether saliency-aware augmentation improves explanation quality.

---

## Directory Structure

```
project/
├── CLAUDE.md                  ← you are here
├── project_proposal.md        ← full research proposal (read this for context)
├── PuzzleMix/                 ← original Puzzle Mix repo (Kim et al., ICML 2020)
│   ├── ...                    ← reference implementation; do not modify directly
├── src/
│   ├── data/                  ← ESC-50 loading, mel-spectrogram conversion
│   ├── models/                ← ResNet-18 adapted for single-channel input
│   ├── augmentation/          ← Puzzle Mix, CutMix, Vanilla Mixup implementations
│   ├── saliency/              ← saliency map computation and evaluation metrics
│   ├── train.py               ← main training loop, all four conditions
│   └── evaluate.py            ← post-training saliency evaluation pipeline
├── experiments/               ← config files, one per training condition
├── results/                   ← saved checkpoints, saliency maps, metric CSVs
└── notebooks/                 ← exploratory analysis and figure generation
```

---

## The Four Training Conditions

All conditions use:
- **Architecture**: ResNet-18, ImageNet-pretrained, first conv modified for 1-channel input, final FC replaced with 50-class head
- **Optimiser**: SGD, momentum=0.9, weight decay=1e-4
- **LR schedule**: Initial LR=0.1, cosine annealing, 200 epochs
- **Batch size**: 64
- **Seeds**: 3 random seeds per condition; report mean ± std

| Condition | Augmentation | Key detail |
|---|---|---|
| `baseline` | Random crop + horizontal flip only | No mixup |
| `vanilla_mixup` | Linear pixel blend | λ ~ Beta(0.4, 0.4) |
| `cutmix` | Random rectangular patch swap | Label mix ∝ patch area |
| `puzzle_mix` | Saliency-aware mask + transport | Kim et al. 2020 hyperparams |

---

## Input Representation

- **Dataset**: ESC-50 (2,000 clips, 50 classes, 40 clips/class, 44.1 kHz)
- **Spectrogram**: 128-bin mel-spectrogram, 25ms window, 10ms hop → shape **(128 × 500)**
- **Preprocessing**: Log compression only. No further normalisation (to avoid confounds with saliency outcomes).
- **Split**: ESC-50's official 5-fold cross-validation; use fold 5 as test, fold 4 as validation, folds 1–3 as training (or standard 80/20 if simpler — document the choice)

---

## Saliency Computation

After training, compute vanilla gradient saliency for all correctly-classified test examples:

```
S(x) = |∇_x f_{ŷ}(x)|
```

- Absolute value elementwise
- Normalise to [0, 1] per sample
- Use the *same* saliency method across all conditions (vanilla gradients) — this is intentional: we ask whether Puzzle Mix improves the very signal it uses internally

---

## Evaluation Metrics (Primary)

### 1. Frequency Band Concentration (FBC)
- Threshold saliency map at top-20% → binary importance mask
- Threshold class-conditional spectral profile at top-20% → binary reference mask
- FBC = IoU between the two masks
- Report: per-class and overall mean ± std across seeds

### 2. Spectral Pointing Game (SPG)
- Identify the single peak saliency cell in each test example
- SPG = proportion of examples where peak falls within top-20% of class profile
- Report: per-class and overall

### 3. Saliency Sharpness (secondary)
- Entropy of normalised saliency map: H(S) = -Σ S(i) log S(i)
- Lower = more concentrated/decisive attributions

### Perceptual Reference Construction
- For each class: average normalised mel-spectrogram energy across all **training** clips of that class
- This is computed from raw audio only — no model involvement
- Represents "what spectral regions are consistently active for this class"
- This is a proxy, not ground truth. Acknowledge limitations for high-variance classes.

---

## Hypotheses to Test

| ID | Hypothesis | What would confirm it | What would falsify it |
|---|---|---|---|
| H1 | Puzzle Mix → higher FBC and SPG than all baselines | Puzzle Mix FBC significantly above CutMix and below | No significant difference; or Puzzle Mix worse |
| H2 | Puzzle Mix → lower saliency entropy (sharper maps) | Puzzle Mix entropy lowest across conditions | Entropy similar or higher |
| H3 | Ordering: Puzzle Mix > CutMix > Vanilla Mixup > Baseline on FBC | Monotonic ordering across conditions | Non-monotonic ordering |
| H4 | Null result: gradient saliency too noisy to improve faithfulness | — | H1 holds | H1 does not hold; report as finding about gradient quality as training signal |

---

## Key Implementation Notes

### Puzzle Mix specifics
- Saliency: L2 norm of input gradient, `‖∇_x ℓ‖₂`, from the **online model** at training time
- Mask optimisation: α-β swap via **pyGCO** (graph cuts library) — understand what this does before using it
- Transport: custom masked optimal transport algorithm (Algorithm 1 in paper) — re-read before implementing
- Adversarial component (Algorithm 2): stochastic adversarial training; this can be included or excluded — document the decision
- Refer to `PuzzleMix/` for the original implementation, but re-implement in `src/augmentation/` so every component is understood

### Libraries
- `torchaudio` for audio loading and mel-spectrogram computation
- `torch.autograd` for gradient saliency (no external attribution library needed for vanilla gradients)
- `pygco` for graph cut optimisation
- `scipy` or `torch` for transport plan computation

### What to avoid
- Do not use attribution libraries (Captum etc.) for the core saliency computation — vanilla gradients should be implemented directly so the mechanism is transparent
- Do not copy-paste the Puzzle Mix repo wholesale; re-implement component by component

---

## Code Quality Constraints

These apply to all code written in this project:

1. **Every function must have a docstring** explaining what it does and why — not just what the inputs/outputs are
2. **No black-box components**: if a library function is used (e.g. from pyGCO), add a comment explaining what it computes and why it's appropriate here
3. **Reproducibility**: all random seeds must be set explicitly; all hyperparameters must be logged, not hardcoded
4. **The student must be able to explain every line** — if a code suggestion would be opaque, break it down and explain the components first

---

## Report Constraints

- **4,000 word cap** — be ruthless about information density
- **XAI framing is mandatory** — every section should connect back to explainability, not just classification performance
- **Negative results are valid** — if H1 doesn't hold, that's a finding, not a failure; frame it as evidence about gradient saliency as a training signal
- Sections: Introduction, Background, Methodology, Experiments, Results, Discussion, Conclusion
- Tone: workshop paper (concise, precise, critical)

---

## How to Use This File (Claude Code Instructions)

When helping with this project:

- **Always read `project_proposal.md`** for detailed context before making suggestions
- **Prioritise understanding over speed** — explain what each component does before writing it
- **Flag XAI connections** whenever relevant — e.g. if a design decision has implications for saliency faithfulness, say so explicitly
- **Respect the hypothesis structure** — when designing experiments, tie them explicitly to H1–H4
- **Question assumptions** — if a methodological choice has a known weakness (e.g. vanilla gradient saliency is noisy), say so and suggest how to account for it
- When writing code, **explain the mathematical operation** it implements (e.g. "this computes the IoU between two binary masks, which we use as the FBC metric because...")
- **Do not write the report prose** — help structure arguments, identify gaps, and give feedback on drafts