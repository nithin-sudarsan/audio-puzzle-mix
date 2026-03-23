"""ESC-50 dataset loader and log-mel spectrogram converter.

This module is responsible for:
  1. Downloading the ESC-50 dataset archive from GitHub if not already present.
  2. Converting each 5-second audio clip to a log-mel spectrogram of shape (128, 500).
  3. Caching the computed spectrograms to disk as .pt tensors so that subsequent
     training runs do not recompute them.
  4. Returning PyTorch Dataset and DataLoader objects for train / val / test splits
     according to the official ESC-50 fold structure.

Fold assignment (fixed, per project specification):
  - Train : folds 1, 2, 3  (1,200 clips)
  - Val   : fold 4          (400 clips)
  - Test  : fold 5          (400 clips)

Spectrogram parameters (per project specification):
  - Sample rate  : 44,100 Hz
  - n_mels       : 128
  - Window length: round(0.025 × 44100) = 1102 samples  ≈ 24.99 ms
                   (Python's round() uses banker's rounding: 1102.5 → 1102)
  - Hop length   : round(0.010 × 44100) = 441 samples   = 10.00 ms exactly
  - n_fft        : 2048  (next power-of-two ≥ win_length; used for FFT efficiency)
  - center=True  : the signal is zero-padded by n_fft//2 on each side before the STFT,
                   which ensures the first and last frames are centred on the signal
                   endpoints rather than starting/ending outside the clip.  With these
                   settings a 220,500-sample clip produces 501 frames; we trim to 500.
  - Compression  : 10 × log10(mel_power + 1e-9)  — dB-scale, clipped to [−80, 0] dB
                   relative to a 1.0-amplitude signal.  No further normalisation is
                   applied (per spec, to avoid introducing confounds with saliency).

Why dB compression?
  Raw mel-filterbank energies span many orders of magnitude.  Log compression
  compresses this dynamic range in a way that matches human auditory perception
  (the ear is also roughly logarithmic in loudness).  The 1e-9 floor prevents
  log(0) for silent frames.  We do NOT z-score or min-max normalise afterwards
  because normalisation would make the saliency maps of different training
  conditions incomparable — differences in saliency magnitude would reflect
  normalisation choices rather than what the model attends to.
"""

import os
import zipfile
import urllib.request
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Spectrogram hyper-parameters — single source of truth for the whole project
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectrogramConfig:
    """All parameters that determine the shape and scale of the spectrograms.

    Frozen so that accidental mutation during training is impossible.
    Every value is documented with its physical meaning.
    """
    sample_rate: int = 44_100      # ESC-50 native sample rate (Hz)
    n_mels: int = 128              # Number of mel filterbank channels
    # Window length in samples.  25 ms × 44100 Hz = 1102.5; banker's rounding → 1102.
    win_length: int = 1_102
    # Hop length in samples.  10 ms × 44100 Hz = 441.0 exactly.
    hop_length: int = 441
    # FFT size: smallest power of two ≥ win_length (1102), which is 2048.
    # A larger n_fft than win_length means the window is zero-padded before FFT,
    # giving finer frequency resolution without changing the time resolution.
    n_fft: int = 2_048
    # Number of time frames after trimming.
    # A 5 s clip = 220,500 samples.  With center=True and hop=441 the STFT
    # produces 501 frames; we trim the last one to get exactly 500.
    n_frames: int = 500
    # Amplitude floor before log compression (prevents log(0))
    amplitude_floor: float = 1e-9
    # dB floor: spectrograms are clipped to [db_floor, 0] dB (0 dB = power 1.0)
    db_floor: float = -80.0


SPEC_CFG = SpectrogramConfig()

# ESC-50 official fold assignments
TRAIN_FOLDS = (1, 2, 3)
VAL_FOLD    = 4
TEST_FOLD   = 5

# Download URL for the ESC-50 archive (GitHub snapshot)
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_ZIP_NAME = "ESC-50-master.zip"
ESC50_DIR_NAME = "ESC-50-master"   # directory name inside the zip


# ---------------------------------------------------------------------------
# Download and integrity helpers
# ---------------------------------------------------------------------------

def _download_esc50(data_root: Path) -> Path:
    """Download and unzip ESC-50 into *data_root* if it is not already present.

    Returns the path to the extracted ESC-50-master directory.

    The download is skipped entirely if the directory already exists, so
    re-running training does not re-fetch the dataset.
    """
    esc50_dir = data_root / ESC50_DIR_NAME

    if esc50_dir.exists():
        return esc50_dir

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / ESC50_ZIP_NAME

    if not zip_path.exists():
        print(f"Downloading ESC-50 from {ESC50_URL} …")
        urllib.request.urlretrieve(ESC50_URL, zip_path)
        print(f"Saved to {zip_path}")

    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)
    print(f"Extracted to {esc50_dir}")

    return esc50_dir


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------

def _build_mel_transform() -> torchaudio.transforms.MelSpectrogram:
    """Construct the torchaudio MelSpectrogram transform with project parameters.

    We build this once and reuse it for all clips.  center=True (the torchaudio
    default) zero-pads the waveform by n_fft//2 on each side before computing
    the STFT, which ensures every input sample contributes to at least one
    analysis frame.  Without centering, the first and last ≈12 ms of each clip
    would be under-represented in the spectrogram.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SPEC_CFG.sample_rate,
        n_fft=SPEC_CFG.n_fft,
        win_length=SPEC_CFG.win_length,
        hop_length=SPEC_CFG.hop_length,
        n_mels=SPEC_CFG.n_mels,
        center=True,      # symmetric zero-padding at signal boundaries
        power=2.0,        # power spectrogram (|STFT|²) before mel filterbank
        norm="slaney",    # area-normalise each mel filterbank triangle so that
                          # low- and high-frequency bands are on comparable scales
        mel_scale="htk",  # HTK formula for mel↔Hz conversion (standard for ESC-50)
    )


def _waveform_to_logmel(waveform: torch.Tensor,
                         mel_transform: torchaudio.transforms.MelSpectrogram,
                         cfg: SpectrogramConfig = SPEC_CFG) -> torch.Tensor:
    """Convert a mono waveform tensor to a log-mel spectrogram.

    Parameters
    ----------
    waveform : Tensor, shape (1, n_samples)
        Mono audio waveform.  If the clip has more than one channel, only the
        first is used (ESC-50 is mono, so this should not occur in practice).
    mel_transform : MelSpectrogram
        Pre-built transform (avoid constructing per clip for efficiency).
    cfg : SpectrogramConfig
        Hyper-parameters; defaults to the module-level SPEC_CFG singleton.

    Returns
    -------
    Tensor, shape (n_mels, n_frames) = (128, 500)
        Log-mel spectrogram in dB, clipped to [db_floor, 0] dB.

    Mathematical operations
    -----------------------
    1. mel = MelFilterbank(|STFT(waveform)|²)       shape: (n_mels, T)
    2. log_mel = 10 × log10(mel + amplitude_floor)  dB relative to power 1.0
    3. log_mel = clip(log_mel, db_floor, 0.0)        removes extreme outliers
    4. log_mel = log_mel[:, :n_frames]               trim from 501 → 500 frames
    """
    if waveform.shape[0] > 1:
        # Downmix to mono by averaging channels
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 1: mel-filterbank power spectrogram, shape (1, n_mels, T)
    mel_power = mel_transform(waveform)  # values are in power units (amplitude²)

    # Step 2: convert to decibels.
    # 10 * log10(x + ε) maps the power to a log scale matching human loudness
    # perception.  The ε floor (1e-9) prevents log(0) for silent frames.
    log_mel = 10.0 * torch.log10(mel_power + cfg.amplitude_floor)

    # Step 3: clip to [db_floor, 0] dB.
    # 0 dB corresponds to a signal with power 1.0 (the maximum after normalised
    # loading from torchaudio).  Values below db_floor are background noise;
    # clipping prevents them from dominating the dynamic range.
    log_mel = torch.clamp(log_mel, min=cfg.db_floor, max=0.0)

    # Step 4: remove the singleton channel dim and trim to n_frames.
    # center=True produces 501 frames for a 220,500-sample clip with hop=441;
    # the 501st frame is centred well past the end of the clip and carries only
    # edge-padding content.  Trimming keeps the shape deterministic at (128, 500).
    log_mel = log_mel.squeeze(0)         # (n_mels, T) — T = 501
    log_mel = log_mel[:, :cfg.n_frames]  # (n_mels, 500)

    return log_mel


# ---------------------------------------------------------------------------
# Spectrogram cache
# ---------------------------------------------------------------------------

def _cache_path(audio_path: Path, cache_dir: Path) -> Path:
    """Return the .pt cache file path for a given audio file.

    The cache filename is derived from the original audio filename so that
    re-downloads or moves do not silently reuse a stale cache.
    """
    return cache_dir / (audio_path.stem + ".pt")


def _load_or_compute_spectrogram(
        audio_path: Path,
        cache_dir: Path,
        mel_transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    """Return the log-mel spectrogram for *audio_path*, using a disk cache.

    On the first call for a given file the spectrogram is computed and saved
    to *cache_dir*.  On subsequent calls the cached tensor is loaded directly,
    avoiding the cost of audio decoding and FFT on every training epoch.

    Parameters
    ----------
    audio_path : Path
        Absolute path to a .wav file in the ESC-50 audio/ directory.
    cache_dir : Path
        Directory where .pt cache files are stored.
    mel_transform : MelSpectrogram
        Pre-built transform shared across all files.

    Returns
    -------
    Tensor, shape (128, 500)
    """
    cached = _cache_path(audio_path, cache_dir)

    if cached.exists():
        return torch.load(cached, weights_only=True)

    # soundfile reads WAV files natively without requiring FFmpeg.
    # It returns a numpy array of shape (n_samples,) for mono or (n_samples, n_ch)
    # for multi-channel, and the integer sample rate.
    # We use soundfile in preference to torchaudio.load because torchaudio >= 2.1
    # routes through torchcodec which requires FFmpeg system libraries.
    samples_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)

    # Ensure shape is (n_samples,) for mono — ESC-50 is always mono
    if samples_np.ndim == 2:
        samples_np = samples_np.mean(axis=1)  # downmix to mono

    # Convert to (1, n_samples) tensor as expected by torchaudio transforms
    waveform = torch.from_numpy(samples_np).unsqueeze(0)

    # Resample if the file's native rate differs from our target.
    # ESC-50 is recorded at 44,100 Hz so this should be a no-op, but we guard
    # against edge cases (e.g. a future dataset variant resampled at 16 kHz).
    if sr != SPEC_CFG.sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=SPEC_CFG.sample_rate
        )
        waveform = resampler(waveform)

    spec = _waveform_to_logmel(waveform, mel_transform)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(spec, cached)

    return spec


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ESC50Dataset(Dataset):
    """PyTorch Dataset for ESC-50 log-mel spectrograms.

    Each item is a (spectrogram, label) pair where:
      - spectrogram is a float32 Tensor of shape (1, 128, 500) — the leading
        dimension is the channel dimension expected by Conv2d in ResNet.
      - label is an int in [0, 49] corresponding to the ESC-50 target column.

    The class also exposes `class_names` (list of 50 strings) and
    `class_to_idx` (dict mapping string → int) for convenience.

    Parameters
    ----------
    esc50_dir : Path
        Root of the extracted ESC-50-master directory (contains meta/ and audio/).
    folds : tuple[int, ...]
        Which ESC-50 folds to include (e.g. (1, 2, 3) for train).
    cache_dir : Path
        Where to write / read pre-computed spectrogram .pt files.
    mel_transform : MelSpectrogram, optional
        If None, a new transform is constructed.  Pass a shared instance when
        constructing multiple splits to avoid repeated initialisation.
    """

    def __init__(
        self,
        esc50_dir: Path,
        folds: tuple,
        cache_dir: Path,
        mel_transform: torchaudio.transforms.MelSpectrogram | None = None,
    ):
        self.esc50_dir = Path(esc50_dir)
        self.cache_dir = Path(cache_dir)
        self.folds = set(folds)

        # Load metadata CSV.  Columns: filename, fold, target, category, ...
        meta_path = self.esc50_dir / "meta" / "esc50.csv"
        meta = pd.read_csv(meta_path)

        # Filter to the requested folds
        meta = meta[meta["fold"].isin(self.folds)].reset_index(drop=True)

        self.filenames = meta["filename"].tolist()
        self.labels    = meta["target"].tolist()     # int, 0–49

        # Build ordered class list from the full metadata (not just the subset)
        # so that class indices are consistent across train/val/test splits.
        full_meta = pd.read_csv(meta_path)
        # Sort by target index to guarantee a stable mapping
        class_df = full_meta[["target", "category"]].drop_duplicates().sort_values("target")
        self.class_names  = class_df["category"].tolist()   # 50 strings
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.mel_transform = mel_transform or _build_mel_transform()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (spectrogram, label) for the idx-th example.

        The spectrogram has an explicit channel dimension (1, 128, 500) so it
        can be passed directly to a Conv2d layer expecting (B, C, H, W).
        """
        audio_path = self.esc50_dir / "audio" / self.filenames[idx]
        spec = _load_or_compute_spectrogram(
            audio_path, self.cache_dir, self.mel_transform
        )
        # Add channel dimension: (128, 500) → (1, 128, 500)
        spec = spec.unsqueeze(0)
        return spec, self.labels[idx]


# ---------------------------------------------------------------------------
# Public interface: build all three DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Download ESC-50 (if needed), precompute spectrograms, and return DataLoaders.

    This is the single entry point used by train.py and evaluate.py.  It handles
    downloading, caching, and DataLoader construction in one call.

    Parameters
    ----------
    data_root : str or Path
        Directory where the ESC-50 archive will be downloaded and extracted.
        Recommended: <project_root>/data/
    batch_size : int
        Number of spectrograms per mini-batch.
    num_workers : int
        Subprocesses for DataLoader prefetching.  Set to 0 to disable
        multiprocessing (useful for debugging).
    seed : int
        Random seed for the training DataLoader's shuffle order.  This seed
        affects data ordering only; model weight initialisation uses its own
        separate seed set in train.py.
    download : bool
        If True, download ESC-50 automatically.  Set to False to raise an
        error if the data is not already present (useful in offline environments).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    class_names : list[str]
        Ordered list of 50 ESC-50 class names (index = label integer).

    Notes on the generator / worker_init_fn
    ----------------------------------------
    PyTorch's DataLoader uses its own internal PRNG for shuffling.  Passing a
    seeded Generator ensures that the shuffle order is reproducible across runs
    with the same seed, which is essential for comparing different training
    conditions fairly.
    """
    data_root = Path(data_root)

    if download:
        esc50_dir = _download_esc50(data_root)
    else:
        esc50_dir = data_root / ESC50_DIR_NAME
        if not esc50_dir.exists():
            raise FileNotFoundError(
                f"ESC-50 not found at {esc50_dir}.  "
                "Pass download=True or download the dataset manually."
            )

    cache_dir = data_root / "spectrogram_cache"

    # Build one shared transform to avoid constructing the mel filterbank 3×
    mel_transform = _build_mel_transform()

    train_dataset = ESC50Dataset(esc50_dir, TRAIN_FOLDS, cache_dir, mel_transform)
    val_dataset   = ESC50Dataset(esc50_dir, (VAL_FOLD,),  cache_dir, mel_transform)
    test_dataset  = ESC50Dataset(esc50_dir, (TEST_FOLD,), cache_dir, mel_transform)

    # Seeded generator for reproducible shuffle order in training
    # (does not affect val/test, which are never shuffled)
    # pin_memory speeds up host→GPU transfers but is only supported with CUDA.
    # On MPS (Apple Silicon) or CPU it is silently ignored or raises a warning.
    use_pin_memory = torch.cuda.is_available()

    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        """Seed each DataLoader worker's numpy/random state independently.

        Without this, all workers share the same numpy random state (inherited
        from the parent process), which can cause correlated augmentation patterns
        within a batch when multiprocessing is active.
        """
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_names


# ---------------------------------------------------------------------------
# Perceptual reference: class-conditional spectral profiles
# ---------------------------------------------------------------------------

def compute_class_profiles(
    train_loader: DataLoader,
    num_classes: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute class-conditional mean log-mel spectrogram for each class.

    This produces the perceptual reference used in the FBC and SPG metrics:
    for each of the 50 ESC-50 classes, we average the (normalised) log-mel
    energy across all training clips of that class.  The result tells us which
    frequency bands and time regions are consistently active for each sound.

    This is computed entirely from raw audio — no model is involved — which is
    why it can serve as an interpretable, model-independent reference.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader over the training split (folds 1–3).
    num_classes : int
        Number of classes (50 for ESC-50).
    device : str
        Computation device.

    Returns
    -------
    profiles : Tensor, shape (num_classes, 128, 500)
        profiles[c] is the mean log-mel spectrogram for class c, with each
        individual spectrogram min-max normalised to [0, 1] before averaging.
        Min-max normalisation within each clip equalises the contribution of
        clips recorded at different loudness levels — we care about the spatial
        pattern of energy (which frequency bands are active) rather than absolute
        amplitude.

    Note on the normalisation choice
    ---------------------------------
    We normalise each clip to [0, 1] before averaging so that a loud dog bark
    and a quiet one contribute equally to the dog-bark profile.  Without this,
    louder clips would dominate the profile, and the reference would reflect
    recording-level variation rather than spectral shape.  This normalisation
    is applied only here (for profile construction) and does not affect the
    spectrograms fed to the model during training.
    """
    sum_specs   = torch.zeros(num_classes, SPEC_CFG.n_mels, SPEC_CFG.n_frames,
                              device=device)
    count       = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for specs, labels in train_loader:
            specs  = specs.to(device)   # (B, 1, 128, 500)
            labels = labels.to(device)

            specs_2d = specs.squeeze(1)  # (B, 128, 500) — drop channel dim

            # Normalise each spectrogram individually to [0, 1].
            # We reshape to (B, -1) to find per-sample min/max, then reshape back.
            flat    = specs_2d.reshape(specs_2d.shape[0], -1)
            s_min   = flat.min(dim=1).values.reshape(-1, 1, 1)
            s_max   = flat.max(dim=1).values.reshape(-1, 1, 1)
            # Avoid division by zero for (pathologically) silent clips
            normed  = (specs_2d - s_min) / (s_max - s_min + 1e-8)

            # Accumulate per class
            for c in range(num_classes):
                mask = (labels == c)
                if mask.any():
                    sum_specs[c] += normed[mask].sum(dim=0)
                    count[c]     += mask.sum()

    # Guard: every class should appear in the training set
    assert (count > 0).all(), "Some classes have zero training examples — check fold assignment."

    profiles = sum_specs / count.reshape(-1, 1, 1)
    return profiles  # (50, 128, 500)
