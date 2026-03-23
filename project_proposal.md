**MPhil Candidate, Explainable Artificial Intelligence, University of Cambridge**

---

## 1. Motivation and Research Question

A central tension in explainable artificial intelligence concerns the relationship between how a model is trained and the quality of the explanations it produces. Most XAI methods — gradient saliency, GradCAM, Integrated Gradients — are applied post-hoc to an already-trained model, with no guarantee that the training procedure has encouraged the model to develop explanations that are faithful, consistent, or human-aligned. Puzzle Mix (Kim et al., 2020) offers an unusual alternative: it places saliency at the centre of the training pipeline itself, using gradient-based importance maps to construct augmented training examples that actively preserve the most discriminative regions of each input. This raises a question that the original paper does not address:

> _Does training with a saliency-aware augmentation method produce models whose own saliency maps are more perceptually faithful — that is, more concentrated on the features that humans recognise as class-defining?_

This project investigates that question by applying Puzzle Mix to a new domain — environmental sound classification using mel-spectrograms — where the connection between model saliency and perceptual salience is both tractable and meaningful. Audio spectrograms are two-dimensional time-frequency representations that admit direct application of image-based augmentation methods, while also having well-understood perceptual structure: each sound class occupies characteristic frequency bands at characteristic moments, and these are knowable independently of any model. This creates a rare opportunity to evaluate saliency quality against an external, interpretable reference rather than relying solely on proxy metrics.

---

## 2. Background

### 2.1 Puzzle Mix

Puzzle Mix constructs mixed training examples by jointly optimising a binary spatial mask **z** and transport plans **Π₀, Π₁** that spatially rearrange each source image before mixing. The mask is solved via α-β swap graph cuts to maximise total exposed saliency subject to a local statistics regularisation penalty, which penalises cuts across regions of high spatial coherence. Formally, the method minimises:

> **E(z) = −saliency exposed by z + λ · spatial discontinuity cost of z**

The saliency signal used is the L2 norm of the input gradient: **s(x) = ‖∇ₓℓ(x, y)‖₂**, computed from the current model at training time. The result is a mixed image that maximally exposes the discriminative content of both source inputs while preserving local structural coherence.

Critically, this means the augmentation is _adaptive_: the mask depends on what the current model finds important. As training progresses and the model's representations sharpen, the saliency signal changes, and so do the masks. This feedback loop between model saliency and training data is the central object of interest in this project.

### 2.2 Saliency Methods and Faithfulness

Gradient-based saliency methods attribute importance to input features by computing how much a small change to each feature would affect the model's output. Vanilla gradient saliency (**∇ₓℓ**) is the simplest such method and serves as the internal signal in Puzzle Mix. It is well known to produce noisy maps that are sensitive to input perturbations and may highlight features that are statistically associated with the class but not causally important for the prediction.

The faithfulness of a saliency map — the degree to which it correctly identifies what the model is actually using — is distinct from its perceptual plausibility. A faithful map tells you what the model uses; a perceptually plausible map highlights what a human would consider class-defining. These can diverge. A key hypothesis of this project is that Puzzle Mix's training procedure may narrow this gap, by consistently rewarding the model for associating class labels with their most saliency-rich features.

### 2.3 Audio Spectrograms as a Transfer Domain

A mel-spectrogram represents an audio clip as a two-dimensional matrix of shape (mel frequency bins × time frames), where each cell encodes the energy of a particular frequency band at a particular moment. The mel scale compresses frequency according to human auditory perception, giving greater resolution to lower frequencies where human hearing is most sensitive.

Environmental sounds have characteristic spectral signatures. A dog bark concentrates energy in low-to-mid frequencies during a brief temporal burst. Rain produces diffuse broadband energy sustained over time. A chainsaw shows strong harmonic structure in mid frequencies with temporal regularity. These class-defining spectral properties are identifiable independently of any model — either from domain knowledge or by computing class-conditional average energy profiles from the dataset. This independence is what makes audio spectrograms a particularly clean domain for evaluating whether model saliency maps capture perceptually meaningful structure.

Puzzle Mix transfers to this domain with minimal modification: the spectrogram is treated as a single-channel image, and the full Puzzle Mix pipeline — saliency computation, graph-cut mask optimisation, transport-based rearrangement, and label mixing — applies directly.

---

## 3. Dataset

This project uses **ESC-50** (Piczak, 2015), a benchmark dataset of 2,000 five-second environmental audio recordings across 50 classes organised into 5 broad categories: animals, natural soundscapes, human non-speech sounds, interior domestic sounds, and exterior urban sounds. Each class contains 40 clips sampled at 44.1 kHz.

ESC-50 is chosen for three reasons. First, its classes have strongly differentiated spectral signatures, making the connection between frequency-band saliency and class identity visually and quantitatively interpretable. Second, its small size permits multiple full training runs on Colab Pro within the project timeline. Third, it is a standard benchmark with reported baselines, permitting sanity-checking of classification performance.

Audio clips will be converted to 128-bin mel-spectrograms using a 25ms window with 10ms hop, producing representations of shape (128 × 500). Log compression is applied to stabilise magnitude. No further preprocessing is applied, to avoid introducing confounds between preprocessing choices and saliency outcomes.

---

## 4. Methodology

### 4.1 Model Architecture

All conditions use a ResNet-18 backbone pretrained on ImageNet, with the first convolutional layer modified to accept single-channel input. The final fully-connected layer is replaced with a 50-class output head. Using the same architecture across all conditions ensures that differences in saliency quality are attributable to the augmentation method rather than to architectural variation.

### 4.2 Training Conditions

Four training conditions are compared:

- **Baseline**: Standard training with no mixup augmentation, using random crop and horizontal flip only
- **Vanilla Mixup**: Linear pixel-level blending of two training examples with mixing coefficient λ ~ Beta(α, α), α = 0.4
- **CutMix**: Random rectangular patch replacement with label mixing proportional to patch area
- **Puzzle Mix**: Saliency-aware mask optimisation with transport-based rearrangement, using the original hyperparameters from Kim et al. (2020)

All conditions use SGD with momentum 0.9, weight decay 1e-4, an initial learning rate of 0.1 with cosine annealing over 200 epochs, and a batch size of 64. These are held constant across conditions. Each condition is run with three random seeds; results are reported as mean ± standard deviation.

### 4.3 Saliency Map Computation

After training, gradient saliency maps are computed for all correctly-classified test examples under each training condition. For a model **f** and input spectrogram **x** with predicted class **ŷ**, the saliency map is:

> **S(x) = |∇ₓ f_{ŷ}(x)|**

where the absolute value is taken elementwise. Maps are normalised to [0,1] per sample. Vanilla gradient saliency is chosen as the evaluation method because it is the same signal used internally by Puzzle Mix during training. This creates a closed, internally consistent evaluation: we ask whether the saliency signal that Puzzle Mix optimises around becomes more perceptually faithful when it is used as a training inductive bias.

### 4.4 Constructing the Perceptual Reference

For each of the 50 classes, a class-conditional spectral profile is constructed by averaging normalised mel-spectrogram energy across all training clips of that class. This profile represents the expected spectral activity for that class — the frequency bands and temporal regions that are consistently active across instances. It is computed entirely from the raw audio, with no model involvement, and serves as an interpretable proxy for perceptual salience: the regions that are consistently energetic across a class are precisely the regions that define its sound.

This is not a ground-truth annotation of human perception, and the proposal does not claim otherwise. It is a data-driven approximation of which spectral regions are class-defining, and its limitations are acknowledged explicitly in the analysis.

### 4.5 Evaluation Metrics

**Frequency Band Concentration (FBC):** For each test example, the saliency map is thresholded at its top-20% values to produce a binary importance mask. The FBC score is the intersection-over-union between this mask and the binarised class-conditional spectral profile (thresholded at its top-20%). A higher FBC indicates that the model's saliency concentrates in the spectrally distinctive regions of the class. This metric is computed per example and averaged across the test set, reported per class and overall.

**Spectral Pointing Game (SPG):** A stricter metric. For each test example, the single highest-saliency cell in the spectrogram is identified. The SPG score is the proportion of test examples for which this peak falls within the top-20% of the class-conditional spectral profile. This is a binary hit/miss metric that is simple to interpret and directly comparable across conditions.

**Saliency Sharpness:** The entropy of the normalised saliency map, measured as **H(S) = −Σ S(i) log S(i)**. Lower entropy indicates a more concentrated, sharper saliency map. This is a secondary metric that captures whether Puzzle Mix training produces more decisive attributions, independent of whether those attributions are perceptually aligned.

---

## 5. Hypotheses

**H1 — Concentration:** Models trained with Puzzle Mix will achieve higher FBC and SPG scores than models trained with CutMix, Vanilla Mixup, or no augmentation. This is because Puzzle Mix's training procedure consistently associates class labels with their most saliency-rich spectral content, reinforcing the model's tendency to attend to class-defining frequency bands.

**H2 — Sharpness:** Models trained with Puzzle Mix will produce lower-entropy saliency maps than all baseline conditions, reflecting the sharper, more localised nature of the features reinforced during training.

**H3 — Ordering:** If H1 holds, the ordering of FBC scores across conditions should follow: Puzzle Mix > CutMix > Vanilla Mixup > Baseline. This ordering reflects the degree to which each method's mixing strategy preserves class-discriminative content. CutMix preserves some spatial structure but randomly, Vanilla Mixup blends globally and destroys local structure entirely, and Baseline receives no mixing signal at all.

**H4 — Negative case:** It is possible that H1 does not hold — that Puzzle Mix's internal saliency signal (vanilla gradients from an online model) is too noisy during early training to reliably identify perceptually meaningful frequency bands, and that the feedback loop amplifies noise rather than signal. This would manifest as no significant difference in FBC between Puzzle Mix and CutMix, or even degraded performance. This outcome would constitute a meaningful finding about the gap between gradient saliency quality and its utility as a training signal, and would be reported and analysed as such.

---

## 6. XAI Connections

This project connects to three core themes in the XAI module.

**Saliency faithfulness.** The project directly tests whether a training procedure that uses saliency as an inductive bias produces more faithful saliency maps post-training. This addresses a gap in the Puzzle Mix paper, which evaluates augmentation quality through classification accuracy and adversarial robustness but does not examine whether the resulting models are more explainable.

**The training-explanation feedback loop.** Most XAI methods assume a clean separation between training and explanation: you train a model, then explain it. Puzzle Mix collapses this separation by making explanation quality a component of the training objective. This raises fundamental questions about whether such feedback loops are beneficial, self-reinforcing, or potentially circular — and the experiments here provide empirical evidence bearing on that question.

**Perceptual alignment.** The project introduces the concept of perceptual faithfulness — whether model saliency aligns with human-interpretable features — and distinguishes it from technical faithfulness. This distinction is underexplored in the literature and is particularly meaningful in audio, where class-defining spectral features are independently knowable and directly perceptible.

---

## 7. Limitations and Scope

Several limitations are acknowledged upfront. The class-conditional spectral profile is a data-driven proxy for perceptual salience, not a human annotation; classes with high within-class spectral variance may produce unreliable profiles. Vanilla gradient saliency is known to be noisy and sensitive to input perturbations; the use of a single saliency method limits generalisability of conclusions. The ResNet-18 architecture was designed for natural images and may not be optimal for spectrograms; however, its widespread use in audio classification makes it a reasonable baseline. ESC-50 is relatively small, which may limit the statistical reliability of class-level results; this is mitigated by reporting confidence intervals and running multiple seeds.

The project does not aim to set a new state of the art in audio classification. Its contribution is interpretive: to use audio spectrograms as a controlled testbed for understanding whether saliency-aware augmentation improves the perceptual faithfulness of the explanations a model produces.
