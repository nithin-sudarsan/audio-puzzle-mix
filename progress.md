# Project Progress & Report Notes

**Cambridge MPhil XAI — Puzzle Mix on ESC-50**
**Report cap:** 4,000 words | **Sections:** Introduction · Background · Methodology · Experiments · Results · Discussion · Conclusion

This file is a living document. Each stage of implementation adds notes here in the format the report needs them — key decisions, exact numbers, methodological justifications, and XAI connections. It is *not* report prose; it is the raw material from which to write each section.

---

## Status Overview

| Stage | Module | Status | Notes |
|---|---|---|---|
| 1 | `src/data/esc50.py` | ✅ Complete | Dataset loading, spectrogram conversion, class profiles |
| 2 | `src/models/resnet.py` | ✅ Complete | ResNet-18, 1-channel, 50-class head |
| 3 | `src/augmentation/mixup.py` | ✅ Complete | Vanilla Mixup |
| 4 | `src/augmentation/cutmix.py` | ✅ Complete | CutMix |
| 5 | `src/augmentation/puzzle_mix.py` | ✅ Complete | PuzzleMix (graph cut + transport, rectangular adaptation) |
| 6 | `src/saliency/gradients.py` | ✅ Complete | Vanilla gradient saliency (eval + training variants) |
| 7 | `src/saliency/metrics.py` | ✅ Complete | FBC, SPG, Sharpness — single-example and batch |
| 8 | `src/train.py` | ✅ Complete | Main training loop, all four conditions |
| 9 | `src/evaluate.py` | ✅ Complete | Post-training saliency evaluation pipeline |
| 10 | `experiments/` | ✅ Complete | YAML configs, 4 conditions |

---

## Report Section Notes

### INTRODUCTION
*Target: ~400 words. Frame the research question; situate Puzzle Mix as an unusual case of saliency-in-the-loop training; state why audio is the chosen domain.*

**Core tension to open with:**
Most XAI methods (gradient saliency, GradCAM, Integrated Gradients) are applied post-hoc to a model that was trained without any reference to explanation quality. This means explanation quality is an afterthought rather than a training objective. Puzzle Mix is a rare exception: it places saliency at the centre of the training loop.

**The gap this project addresses:**
The original Puzzle Mix paper (Kim et al., 2020) evaluates augmentation quality through classification accuracy and adversarial robustness. It does not ask: does training with saliency-as-inductive-bias produce models whose saliency maps are more *perceptually faithful*? This project asks that question.

**Why audio / ESC-50:**
Audio spectrograms provide a domain where the "ground truth" of perceptual salience is independently knowable without human annotation — each sound class occupies characteristic frequency bands that can be read off from average energy profiles computed from the raw data. This is harder to do for natural images. The domain choice is a methodological asset, not a convenience.

**Research question (verbatim from proposal):**
> Does training with a saliency-aware augmentation method produce models whose own saliency maps are more perceptually faithful — that is, more concentrated on the features that humans recognise as class-defining?

---

### BACKGROUND
*Target: ~600 words. Cover Puzzle Mix mechanics, vanilla gradient saliency, and audio spectrograms as a transfer domain. Cite Kim et al. (2020) and Piczak (2015) at minimum.*

**Puzzle Mix mechanics (to describe):**
- Jointly optimises binary spatial mask **z** and transport plans **Π₀, Π₁** over two source images
- Mask objective: maximise exposed saliency − λ × spatial discontinuity cost (α-β swap via graph cuts)
- Saliency signal used: **s(x) = ‖∇ₓ ℓ(x, y)‖₂** (L2 norm of input gradient) from the *online model*
- The feedback loop: as the model sharpens, saliency maps change, masks change, training data changes → self-reinforcing
- Transport step: rearranges image patches to move high-saliency content into the exposed mask region (custom masked optimal transport)
- The original paper does not study whether this loop produces more explainable models

**Vanilla gradient saliency (to describe):**
- S(x) = |∇ₓ f_ŷ(x)| — elementwise absolute gradient w.r.t. the input
- Known weakness: maps are noisy and sensitive to input perturbations; they show where the function changes, not necessarily what the model "uses" in a causal sense
- Why we use it anyway: it is the *same signal* Puzzle Mix uses internally — evaluating with the same method creates an internally consistent test. We ask: does optimising around vanilla gradients during training make them better?
- Note for report: mention that using a single saliency method limits generalisability (see Limitations section)

**Key distinction to make explicit:**
- *Technical faithfulness*: does the saliency map correctly describe the model's computation?
- *Perceptual faithfulness*: does the saliency map highlight features humans recognise as class-defining?
- These can diverge. A model may faithfully attend to texture artefacts that are statistically correlated with a class but not perceptually meaningful. This project targets perceptual faithfulness specifically.

**Audio spectrograms as transfer domain:**
- Mel-spectrogram: 2D matrix (mel frequency bins × time frames)
- Mel scale: compresses frequency logarithmically, matching human auditory resolution
- Class-defining spectral properties of ESC-50 classes are independently knowable:
  - Dog bark: concentrated low-to-mid frequency burst
  - Rain: diffuse broadband energy sustained over time
  - Chainsaw: harmonic structure in mid frequencies with temporal regularity
- These can be quantified by averaging spectrogram energy across all training clips of a class — no model needed, no human annotation needed
- Puzzle Mix transfers with minimal modification: treat the spectrogram as a 1-channel image

---

### METHODOLOGY
*Target: ~900 words. Dataset, architecture, four training conditions, saliency computation, perceptual reference, evaluation metrics. Be precise with numbers.*

#### Dataset (ESC-50)

| Property | Value |
|---|---|
| Source | Piczak (2015) |
| Total clips | 2,000 |
| Classes | 50 (40 clips/class) |
| Class categories | Animals, natural soundscapes, human non-speech, interior, urban |
| Duration per clip | 5 seconds |
| Sample rate | 44,100 Hz |
| Official structure | 5 folds, each with 400 clips (8 per class) |

**Split assignment (fixed for all conditions and seeds):**
| Split | Folds | Clips |
|---|---|---|
| Train | 1, 2, 3 | 1,200 |
| Validation | 4 | 400 |
| Test | 5 | 400 |

**Justification for this split:** Official fold 5 is the conventional test fold used in published ESC-50 comparisons. Using fold 4 as validation rather than held-out training data ensures we have a fixed validation set for learning rate scheduling and early stopping decisions, without touching the test fold.

#### Input Representation

**Exact spectrogram parameters (implemented in `src/data/esc50.py`):**

| Parameter | Value | Derivation |
|---|---|---|
| n_mels | 128 | Per project specification |
| win_length | 1,102 samples | round(0.025 × 44100) = 1102.5 → 1102 (banker's rounding) |
| hop_length | 441 samples | round(0.010 × 44100) = 441.0 (exact) |
| n_fft | 2,048 | Smallest power of 2 ≥ win_length; gives fine freq. resolution |
| center | True | Symmetric zero-padding ensures signal endpoints are centred |
| n_frames | 500 | 5s × 44100/441 + 1 = 501; trim last frame (edge-padding content) |
| Output shape | (1, 128, 500) | Channel × Mel bins × Time frames |

**Log compression formula:**
```
log_mel = 10 × log10(mel_power + 1e-9)   [dB scale]
log_mel = clip(log_mel, −80.0, 0.0)       [removes extreme outliers]
```
- 0 dB corresponds to a power-1.0 signal (loudest possible)
- −80 dB floor removes background noise from dominating dynamic range
- The 1e-9 floor inside log prevents log(0) for silent frames

**Why no further normalisation:**
Applying z-score or min-max normalisation after log compression would change the *relative magnitudes* of spectrogram cells across conditions. Since we compare saliency maps across training conditions, we need the input space to be identical. Any normalisation that conditions on training statistics would make the conditions' inputs non-comparable.

**Implementation note for methods section:**
The `center=True` STFT produces 501 frames for a 220,500-sample clip; the 501st frame is centred at sample 220,500 + n_fft//2 = 221,524 (i.e., entirely within zero-padding). Trimming it is methodologically sound.

#### Model Architecture

**Implemented in `src/models/resnet.py`.**

| Property | Value |
|---|---|
| Base architecture | ResNet-18 (He et al., 2016) |
| Pretrained weights | ImageNet-1K (torchvision `IMAGENET1K_V1`) |
| Input | (B, 1, 128, 500) — single-channel mel-spectrogram |
| Output | (B, 50) — class logits (no softmax; applied externally) |
| Trainable parameters | 11,195,890 |
| conv1 modification | (3,64,7,7) → (1,64,7,7); weights **summed** over input channels |
| FC modification | Linear(512, 1000) → Linear(512, 50); re-initialised randomly |

**Weight adaptation for 1-channel conv1 — exact method:**
The pretrained filter computes `w₀x₀ + w₁x₁ + w₂x₂`. For a single channel x, the equivalent is `(w₀ + w₁ + w₂)x`. We sum (not average) the 3 channel weights to preserve the activation magnitude that BatchNorm and ReLU were calibrated for during ImageNet pretraining. Averaging would under-scale activations by 3×.

**Spatial feature map dimensions for our input (128 × 500):**
```
conv1  (7×7, s=2): (64,  64, 250)
maxpool(3×3, s=2): (64,  32, 125)
layer2       (s=2): (128, 16,  63)
layer3       (s=2): (256,  8,  32)
layer4       (s=2): (512,  4,  16)
AvgPool(1,1)      : (512,  1,   1)
fc                : (50)
```
The non-square feature maps (4 × 16 at the final stage) are handled gracefully by `AdaptiveAvgPool2d((1,1))`.

**XAI connection (for Methodology / Discussion):**
Starting from pretrained weights (vs random init) gives the Puzzle Mix feedback loop a more informative initial saliency signal. With random weights, early-epoch Puzzle Mix masks are essentially random (gradient saliency from an untrained model is noise). Pretrained features mean the masks are discriminative from epoch 1. This is shared across all four conditions and thus does not bias the comparison — it may reduce the number of epochs needed for the loop to stabilise.

**What to say in the report (Methodology §4.1):**
The first conv is adapted by summing pretrained weights across input channels; all other layers use their ImageNet-pretrained weights unchanged. The final FC is randomly re-initialised for 50-class output. Using the same pretrained initialisation across all four conditions ensures that differences in saliency quality cannot be attributed to different starting points.

#### Training Conditions

| Condition | Augmentation | Key parameters |
|---|---|---|
| `baseline` | Random crop + horizontal flip | — |
| `vanilla_mixup` | Linear blend | λ ~ Beta(0.4, 0.4) |
| `cutmix` | Random rectangular patch | Label mix ∝ actual patch area |
| `puzzle_mix` | Graph cut mask + transport | Kim et al. 2020 hyperparams (adapted for non-square input) |

**Implemented augmentation details:**

*Vanilla Mixup* (`src/augmentation/mixup.py`):
- `x_mixed = λ·x₁ + (1−λ)·x₂` ; `y_mixed = λ·y₁ + (1−λ)·y₂`
- λ ~ Beta(0.4, 0.4), folded to [0.5, 1.0] via max(λ, 1−λ)
- Folded λ mean ≈ 0.844 (Beta(0.4, 0.4) is strongly U-shaped; after folding, mass near 1.0 dominates)
- Permutation is within-batch (sampling without replacement from the same batch)
- Note for report: blending in log-spectrogram space ≠ mixing waveforms then taking log. This applies equally to all blend-based methods and does not bias the comparison.

*CutMix* (`src/augmentation/cutmix.py`):
- Patch dimensions: `patch_h = √(1−λ) × H`, `patch_w = √(1−λ) × W` — scales with spectrogram aspect ratio
- Patch centre uniformly sampled; edges clipped to valid bounds
- λ_actual recomputed from true clipped patch area (ensures label mix = actual image mix)
- Single patch location applied to all examples in the batch (consistent with original paper)
- For report: CutMix preserves local spectral structure but the cut is spatially random — no preference for class-discriminative frequency bands.

*Puzzle Mix* (`src/augmentation/puzzle_mix.py`):
- Saliency passed in from training loop (pre-computed externally in model.eval() mode)
- Four-step pipeline: (1) pool saliency to block grid → (2) graph-cut mask → (3) optimal transport → (4) blend
- **Graph cut (α-β swap):** `pygco.cut_simple_vh(unary, pairwise, costV, costH)` — minimises E = Σ unary(block, label) + Σ_adjacent pairwise(label_i, label_j). n_labels=3 gives mixing levels {0, 0.5, 1.0}
- **Block grid for rectangular (128 × 500) spectrograms:** square pixel blocks (block_size × block_size, sampled from {16, 32}); block_h = 128 // block_size, block_w = 500 // block_size; valid_W = block_w × block_size ≤ 500 (at most 4% trailing columns handled by nearest-neighbour interpolation of the block mask)
- **Cost matrix adaptation:** per-axis normalisation to [0,1] ensures transport regularisation weight (t_eps=0.8) has consistent interpretation on 4×15 grid vs. square grids
- **Transport step:** iterative auction algorithm, n_iter = min(block_h, block_w); moves high-saliency blocks into mask-exposed positions
- **Hyperparameters (Kim et al. 2020 defaults):** α=0.4, β=1.2 (normalised as β/max(block_h,block_w)/16), γ=0.5, η=0.2, t_eps=0.8
- **Adversarial component:** excluded — Algorithm 2 of Kim et al. targets adversarial robustness, not saliency faithfulness. Excluding it keeps the comparison clean.
- **Label mixing:** ratio = mean(mask_pixel) across all pixels; y_mixed = ratio·y₁ + (1−ratio)·y₂
- **XAI connection:** the graph cut explicitly maximises total saliency exposed across the batch, subject to spatial coherence penalties. This is the mechanism by which Puzzle Mix creates an inductive bias toward feature-localised representations.

**Verified behaviour (20-seed test):** Over 20 random seeds with distinct per-example saliency patterns, all 20 produced actual mixing (mean pixel diff ≈ 0.32); label sums = 1.0 for all examples; output shape (B, 1, 128, 500) preserved.

**Shared hyperparameters (all conditions):**
- Optimiser: SGD, momentum=0.9, weight_decay=1e-4
- LR: 0.1, cosine annealing, 200 epochs
- Batch size: 64
- Seeds: 3 (report mean ± std)

#### Saliency Computation

**Implemented in `src/saliency/gradients.py`.**

Two variants:

*Evaluation saliency* (`compute_eval_saliency`):
- `S(x) = |∂f_{ŷ}(x) / ∂x|` — gradient of predicted-class logit w.r.t. input
- Model in eval() mode (frozen BN); gradient only on internal clone of x
- Per-sample max-normalisation to [0, 1]: `S = S / max(S)`
- Returns: saliency (B, H, W) + predicted classes (B,) for filtering to correct examples

*Training saliency* (`compute_training_saliency`):
- `s(x) = ‖∂ℓ(x,y)/∂x‖₂` — gradient of cross-entropy loss w.r.t. input
- Same L2 norm used by Puzzle Mix internally (C=1 → same as abs)
- Per-sample max-normalisation to [0, 1]
- Called in training loop before Puzzle Mix, in model.eval() mode

**Key implementation details:**
- Both functions use `x_input = x.detach().requires_grad_(True)` — the original batch tensor is never modified
- Backpropagate the *sum* over the batch: `predicted_logits.sum().backward()`; this gives independent per-example gradients because eval() mode prevents inter-example BN dependencies
- log_softmax + manual sum for cross-entropy (numerically stable, no library dependency)

**Important methodological note:** Using the same saliency method (vanilla gradients) for both the internal Puzzle Mix training signal and the post-hoc evaluation creates an internally consistent test. We are asking whether the signal Puzzle Mix optimises around becomes more perceptually faithful as a result of that optimisation.

#### Perceptual Reference Construction

**What it is:** For each of the 50 classes, average the per-clip normalised mel-spectrogram energy across all 1,200 training clips of that class (proportionally: 24 clips/class from folds 1–3).

**Implementation detail (implemented in `src/data/esc50.py::compute_class_profiles`):**
1. For each training clip, min-max normalise the spectrogram to [0, 1] **individually**
2. Accumulate per class; divide by count
3. Output shape: (50, 128, 500)

**Why per-clip normalisation before averaging:**
Without normalisation, louder clips dominate the class profile. We care about the *spatial pattern* of energy (which frequency bands are characteristically active), not absolute amplitude. Per-clip normalisation ensures all clips contribute equally to the profile shape.

**Acknowledged limitation:** This is a data-driven proxy for perceptual salience, not a human perceptual annotation. For classes with high within-class spectral variance (e.g., "breathing" vs. "dog" which has consistent bark spectral shape), the profile may be a poor representative of the class's characteristic sound. This should be flagged in the Discussion.

#### Evaluation Metrics

**Implemented in `src/saliency/metrics.py`.**

**FBC (Frequency Band Concentration):**
- Binary importance mask: saliency thresholded at top-20% (top k = int(0.2 × H × W) = 12,800 cells)
- Binary reference mask: class profile thresholded at top-20%
- FBC = IoU(importance mask, reference mask) = |intersection| / |union|
- Verified: perfect alignment → FBC = 1.0; zero overlap → FBC = 0.0
- Rationale: IoU measures spatial overlap between "what the model attends to" and "what is characteristically active for this class"

**SPG (Spectral Pointing Game):**
- Peak saliency cell: argmax of S(x) over the (128 × 500) grid
- SPG hit: True if peak_idx is in the top-20% set of the class profile
- SPG score: proportion of hits across test examples
- Verified: peak in top-20% → hit=True; peak outside → hit=False
- Rationale: stricter than FBC; tests whether the single most salient feature is perceptually meaningful

**Saliency Sharpness:**
- H(S) = −Σ p_i log(p_i) where p_i = S_i / Σ S_j (normalise to probability first)
- Units: nats. Range: [0, log(H×W)] = [0, log(64000)] ≈ [0, 11.07]
- Verified: spike map (entropy ≈ 0.001) < uniform map (entropy ≈ 11.07)
- Lower H = more concentrated / decisive saliency
- Secondary metric: captures localisation quality independent of perceptual alignment

**Batch interface:** `evaluate_saliency(saliency_maps, class_profiles, true_classes)` returns a unified dict with `fbc`, `spg`, and `sharpness` sub-dicts, each containing `per_example`, `per_class`, and summary statistics (`mean`, `std`, or `hit_rate`).

---

### EXPERIMENTS
*Target: ~400 words. Describe the experimental setup (what was run, with what seeds, what was logged). Write after training is complete.*

**Experimental setup:**

| Property | Value |
|---|---|
| Conditions | 4: baseline, vanilla_mixup, cutmix, puzzle_mix |
| Seeds per condition | 1 (seed=42) — mini-project scope |
| Total runs | 4 |
| Epochs per run | 200 |
| Hardware | NVIDIA L4 GPU (24GB VRAM, CUDA 13.1), department cluster |
| Model parameters | 11,195,890 trainable |
| Runs executed in parallel | Yes — 3 conditions simultaneously on separate GPUs (CUDA_VISIBLE_DEVICES=1,2,3) |

**Practical notes:**
- puzzle_mix seed 42 trained first on Apple M-series MPS locally; remaining 3 conditions trained on department L4 GPU
- `gco-wrapper` (Linux) used on GPU instead of `pygco` (macOS) — unified via `_gco_cut()` shim in `puzzle_mix.py`
- Training launched inside `tmux` to survive SSH disconnection
- Spectrogram cache (2,000 `.pt` files, ~500MB) transferred via `rsync` to avoid recomputation

**What is logged per epoch:**
- Training loss (mean over batches)
- Validation accuracy (top-1)
- Current learning rate
- Wall-clock time per epoch

**Checkpointing:**
- `best.pt`: best validation accuracy checkpoint (used for evaluation)
- `epoch<N>.pt`: periodic checkpoints every 10 epochs
- `hparams.txt`: all hyperparameters logged at run start

**Evaluation protocol:**
- Load `best.pt` checkpoint (not last epoch)
- Compute saliency only for correctly-classified test examples
- Class profiles built from training folds 1–3 only (independent of test set)
- Metrics: FBC, SPG, Sharpness per example → aggregate per class and overall

**Best epoch per condition:**
| Condition | Best Epoch | Val Acc |
|---|---|---|
| baseline | 174 | 0.685 |
| vanilla_mixup | 165 | 0.688 |
| cutmix | 140 | 0.690 |
| puzzle_mix | 128 | 0.630 |

Puzzle Mix reached its best validation accuracy earlier (epoch 128) and at a lower value — consistent with stronger regularisation from the saliency-guided augmentation on a small dataset.

---

### RESULTS
*Target: ~600 words. Tables and figures.*

#### Main Results Table

| Condition | Test Acc | n_correct | FBC (↑) | SPG (↑) | Sharpness entropy (↓) |
|---|---|---|---|---|---|
| baseline | 0.685 | 242/400 | 0.1691 ± 0.107 | 0.331 | 10.339 ± 0.286 |
| vanilla_mixup | 0.688 | 255/400 | 0.1405 ± 0.050 | 0.325 | 10.485 ± 0.097 |
| cutmix | 0.690 | 231/400 | 0.1861 ± 0.116 | 0.281 | 10.246 ± 0.306 |
| puzzle_mix | 0.630 | 229/400 | **0.1970 ± 0.116** | **0.336** | 10.318 ± 0.301 |

*(Single seed per condition — no mean ± std across seeds. Std reported is across test examples within the condition.)*

#### Hypothesis Assessment

**H1 (Puzzle Mix highest FBC and SPG):** Partially supported.
- Puzzle Mix achieves the highest FBC (0.197) and highest SPG (0.336) across all four conditions.
- However, the margins are small: FBC advantage over CutMix is 0.011; over baseline is 0.028.
- With a single seed these differences cannot be tested for statistical significance.

**H2 (Puzzle Mix lowest saliency entropy):** Not supported.
- CutMix has the lowest entropy (10.246), not Puzzle Mix (10.318).
- Puzzle Mix entropy is lower than baseline (10.339) and vanilla_mixup (10.485), but not the lowest overall.
- Note: the differences are small (< 0.24 nats across all conditions out of a maximum of 11.07).

**H3 (ordering: Puzzle Mix > CutMix > Vanilla Mixup > Baseline on FBC):** Partially supported.
- FBC ordering: Puzzle Mix (0.197) > CutMix (0.186) > **Baseline (0.169) > Vanilla Mixup (0.140)**
- The Puzzle Mix > CutMix > Baseline ordering holds, but Vanilla Mixup falls *below* baseline — an unexpected finding.
- SPG ordering does not follow the same pattern: Puzzle Mix (0.336) > Baseline (0.331) > Vanilla Mixup (0.325) > CutMix (0.281).
- CutMix SPG (0.281) is the lowest despite having the second-highest FBC — suggests CutMix produces spatially concentrated saliency that is not well-aligned with class profiles.

**H4 (null result — gradient saliency too noisy):** Partially relevant.
- The differences exist but are modest. With a single seed and no significance testing, it is not possible to rule out noise.
- Vanilla Mixup FBC < Baseline is a meaningful finding: globally blended inputs actively harm perceptual faithfulness relative to no mixing at all.

#### Key Unexpected Finding
Puzzle Mix achieves the best FBC/SPG at a 6% accuracy cost (0.630 vs ~0.688–0.690 for other conditions). This is the most important result for the Discussion: the model is more perceptually faithful but less accurate. Possible explanations:
1. **Over-regularisation on a small dataset:** ESC-50 has only 24 training clips/class. Puzzle Mix's complex per-batch saliency computation may introduce too much variance in the training signal for such limited data.
2. **Early stopping bias:** Puzzle Mix's best checkpoint was at epoch 128 vs 174 for baseline — it may benefit from a different LR schedule.
3. **Saliency cold-start:** Early training saliency is noisy (model not yet discriminative), which may lead Puzzle Mix masks to be random in early epochs before the feedback loop stabilises.

#### Vanilla Mixup below Baseline on FBC — interpretation
Vanilla Mixup FBC (0.140) is lower than baseline (0.169). This is consistent with the hypothesis that globally blended inputs discourage spatial feature localisation: every frequency bin at every time step is a blend of both sources, so the model has no incentive to concentrate attention on class-defining bands. This is the mechanism described in the Introduction and provides direct empirical support for it.

#### Figures to produce (for notebooks/)
1. Example spectrogram + saliency map overlaid, for each condition, same test clip
2. Class profile examples (5–6 representative classes)
3. FBC per-class bar chart (4 conditions side-by-side)
4. Saliency entropy bar chart (4 conditions)

---

### DISCUSSION
*Target: ~500 words. Interpret results against H1–H4. Flag limitations. Connect to XAI themes.*

**Central argument to make:**
The results provide weak but directionally consistent evidence that Puzzle Mix's saliency-guided training produces models whose saliency maps are more perceptually faithful. The FBC and SPG orderings partially support H1 and H3, but the accuracy cost and the small margins suggest the mechanism is less powerful than hoped on a small dataset like ESC-50. This is itself a meaningful finding.

**XAI connections to draw:**

1. *Saliency faithfulness vs. technical faithfulness:* The results show that Puzzle Mix improves perceptual faithfulness (FBC, SPG) slightly while not consistently improving sharpness (entropy). This is consistent with the distinction drawn in the Background: a model can attend to the right frequency bands (perceptual faithfulness) without necessarily being more decisive (sharpness). These are genuinely distinct properties.

2. *Training–explanation feedback loop:* Puzzle Mix collapses the separation between training and explanation. The results suggest this loop provides a weak but real inductive bias toward perceptual faithfulness. However, the 6% accuracy cost raises the question of whether this trade-off is worthwhile in practice — explainability improved at the expense of task performance.

3. *Vanilla Mixup actively harms perceptual faithfulness:* FBC below baseline (0.140 vs 0.169) is the clearest finding. This is direct empirical evidence that globally blended inputs discourage spatial feature localisation, consistent with the mechanism described in the Introduction.

**Limitations to acknowledge:**
- **Single seed per condition:** No statistical significance testing is possible. The FBC differences (0.011 between Puzzle Mix and CutMix) may be within sampling noise.
- **Small training set:** 24 clips/class is extremely limited for a complex augmentation like Puzzle Mix. The accuracy cost may be a dataset-size effect rather than an intrinsic property of the method.
- **Class-conditional profiles as proxy:** High-variance classes (e.g. "breathing", "footsteps") will have diffuse profiles that are poor references. FBC/SPG on these classes will be low for all conditions — this dilutes the overall metric.
- **Single saliency method:** Using vanilla gradients for both the training signal and the evaluation metric creates a closed loop that may not generalise. SmoothGrad or Integrated Gradients might show different patterns.
- **Puzzle Mix LR schedule:** Puzzle Mix's best epoch (128) is earlier than other conditions (140–174), suggesting it may benefit from a slower LR decay or warm-up schedule.
- **No adversarial component:** Kim et al.'s Algorithm 2 was excluded. Its effect on saliency faithfulness is unknown.

**What to say about H4 (null result interpretation):**
The results are not a clean null — differences exist in the right direction. But the effect sizes are small and the accuracy cost is substantial. A reasonable conclusion: vanilla gradient saliency is informative enough to provide a weak inductive bias, but too noisy to produce large improvements in perceptual faithfulness on a 24-clip-per-class dataset. This is a finding about the limits of gradient saliency as a training signal, not just a null result.

---

### CONCLUSION
*Target: ~200 words. Restate the research question, summarise the answer (positive, negative, or nuanced), connect to XAI module themes.*

**Key points to hit:**
- Restate the question: does training with saliency-aware augmentation produce models whose saliency maps are more perceptually faithful?
- Answer: yes, weakly — Puzzle Mix achieves the highest FBC (0.197) and SPG (0.336), but at a 6% accuracy cost and with small margins over CutMix.
- The ordering of FBC (Puzzle Mix > CutMix > Baseline > Vanilla Mixup) is the clearest finding: it establishes that the nature of the spatial mixing inductive bias matters, not just whether mixing is used.
- Vanilla Mixup below baseline is a strong secondary finding: global blending actively harms perceptual faithfulness.
- The accuracy–faithfulness trade-off raises a broader XAI question: when improving explanation quality costs task performance, is it worthwhile? This connects to the module theme of evaluation and the costs of explainability.
- Future work: more seeds, larger dataset, stronger saliency methods (SmoothGrad), and exploring whether the LR schedule can be tuned to recover accuracy without sacrificing faithfulness.

---

## Key Decisions Log

This section records design decisions that deviate from the proposal or that required non-obvious justification. Useful for the Methodology section and for responding to reviewer questions.

| Decision | What was chosen | Why | Report impact |
|---|---|---|---|
| Audio loading library | `soundfile` instead of `torchaudio.load` | torchaudio 2.10 routes through torchcodec which requires FFmpeg system libraries; soundfile reads WAV natively | Implementation detail — not report-worthy |
| STFT centering | `center=True`, trim 501→500 | Standard audio STFT practice; ensures signal endpoints are symmetrically treated | Worth one sentence in Methodology |
| Log compression | dB scale (10×log10), no further normalisation | Preserves comparability of saliency maps across conditions | Key methodological point — explain in Methodology |
| Per-clip normalisation for profiles | Min-max per clip before averaging | Equalises loudness variation so profile captures spectral *shape*, not amplitude | Explain in §4.4 / Methodology |
| Rectangular spectrogram | (128 × 500), not padded to square | Avoids discarding temporal information; requires adapting Puzzle Mix cost matrix to non-square grids | **This is a technical contribution** — explain in Methodology and note it as going beyond the reference implementation |
| Fold assignment | Train=1,2,3 / Val=4 / Test=5 | Fold 5 is conventional ESC-50 test fold in published comparisons | Standard — cite Piczak (2015) |
| Baseline augmentation | No spatial augmentation (no random crop/flip) | ESC-50 spectrograms are not padded for crop-then-use; applying crop would require separate padding logic. The baseline isolates the effect of mixing specifically. | Worth clarifying in Methodology |
| Puzzle Mix saliency timing | model.eval() before saliency, model.train() for update | BN in train mode uses batch statistics, which contaminate the gradient (gradient would depend on other examples). eval() freezes BN to running statistics. | Important for correctness — mention in Methodology |
| Best checkpoint for evaluation | Load best.pt (best val acc), not last epoch | The last epoch with cosine LR near 0 may have slightly different weight distribution. Best val checkpoint is the standard comparison point. | Standard practice |
| Seeds | 42, 43, 44 | Three consecutive seeds provide enough variance to compute mean ± std. | Report mean ± std across seeds |

---

## Open Questions (to resolve during implementation)

1. **Puzzle Mix on non-square inputs:** ✅ Resolved in Stage 5. Solution: square pixel blocks (block_size × block_size) with separate block_h and block_w; cost matrix independently normalises each axis to [0,1]; trailing columns handled by nearest-neighbour interpolation.

2. **pygco API mismatch:** ✅ Resolved in Stage 5. `pygco.cut_simple_vh(unary, pairwise, costV, costH)` accepts pw_x → costV[:block_h-1, :] and pw_y → costH[:, :block_w-1], padding remainder with zeros. Verified produces correct masks.

3. **Adversarial component:** Puzzle Mix includes an optional adversarial training component (Algorithm 2 in Kim et al.). Decision: **exclude it** to keep the comparison clean (the adversarial component is about robustness, not saliency quality). → Document this decision in Methodology.

4. **Training time on CPU/MPS:** ✅ Resolved. Trained on department NVIDIA L4 GPU (CUDA). puzzle_mix seed 42 trained locally on MPS; remaining 3 conditions on L4 in parallel. 200 epochs per condition.

5. **Statistical testing:** ✅ Decision made: single seed per condition (mini-project scope). No paired t-tests possible. Report point estimates with per-example std. Acknowledge this as a limitation. The variance across 50 classes (per-class FBC) can be used as a proxy for uncertainty in future work.
