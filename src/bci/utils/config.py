"""Centralized configuration management using YAML files and dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Project root is 3 levels up from this file: src/bci/utils/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Filter bank sub-bands (mu + beta, 8–32 Hz in 4 Hz steps)
# Used by FBCSPFeatureExtractor and FBRiemannianFeatureExtractor.
# ---------------------------------------------------------------------------
FILTER_BANK_BANDS: list[tuple[float, float]] = [
    (8.0, 12.0),  # mu / lower-alpha
    (12.0, 16.0),  # lower-beta
    (16.0, 20.0),  # mid-beta
    (20.0, 24.0),  # mid-beta
    (24.0, 28.0),  # upper-beta
    (28.0, 32.0),  # upper-beta
]


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    name: str = "bci_iv2a"
    data_dir: str = str(DATA_DIR / "raw")
    subjects: list[int] = field(default_factory=lambda: list(range(1, 10)))
    classes: list[str] = field(default_factory=lambda: ["left_hand", "right_hand"])
    # Channel selection: motor cortex channels
    channels: list[str] = field(
        default_factory=lambda: [
            "FC5",
            "FC3",
            "FC1",
            "FC2",
            "FC4",
            "FC6",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
        ]
    )


@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing."""

    # Filtering
    l_freq: float = 4.0
    h_freq: float = 40.0

    # Notch filter
    apply_notch: bool = True
    notch_freq: float = 50.0

    # Trial rejection (peak-to-peak amplitude threshold)
    apply_trial_rejection: bool = True
    rejection_threshold_uv: float = 100.0  # μV

    # Current Source Density (surface Laplacian)
    apply_csd: bool = False  # off by default; requires electrode positions

    # Euclidean alignment (re-center covariance per subject)
    apply_euclidean_alignment: bool = True

    # Time-window crop (applied after MOABB epoching)
    tmin_crop: float = 0.5  # seconds from cue onset
    tmax_crop: float = 3.5  # → 3.0 s window, 384 samples at 128 Hz

    # Z-score normalization per channel per trial
    apply_zscore: bool = True

    # Legacy fields kept for backward compatibility
    apply_ica: bool = False
    n_ica_components: int | None = None
    ica_method: str = "fastica"
    tmin: float = 0.0
    tmax: float = 4.0
    resample_freq: float | None = 128.0
    normalize: bool = True


@dataclass
class SpectrogramConfig:
    """Configuration for CWT spectrogram generation."""

    wavelet: str = "morl"
    freq_min: float = 4.0
    freq_max: float = 40.0
    n_freqs: int = 64
    image_size: tuple[int, int] = (224, 224)
    # How to compose channels into an image
    # "rgb_c3_cz_c4" = map C3→R, Cz→G, C4→B (legacy 3-channel)
    # "mosaic"       = tile multiple channel spectrograms into a grid
    # "multichannel" = produce one image per selected channel (N-channel ViT input)
    channel_mode: str = "rgb_c3_cz_c4"
    # Channels to use in "multichannel" mode (9 motor-cortex channels)
    spectrogram_channels: list[str] = field(
        default_factory=lambda: ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4", "CP3", "CP4"]
    )
    # Normalisation applied after CWT magnitude computation
    # "per_channel"  = normalise each channel spectrogram independently (legacy)
    # "joint"        = normalise across all channels jointly (preserves laterality)
    normalize_mode: str = "joint"
    # Whether to apply ImageNet mean/std normalisation after [0,1] scaling
    apply_imagenet_norm: bool = True
    output_dir: str = str(DATA_DIR / "spectrograms")


@dataclass
class AugmentationConfig:
    """Configuration for EEG and spectrogram data augmentation."""

    # EEG-level augmentation (applied to raw epochs before CWT)
    apply_gaussian_noise: bool = True
    gaussian_noise_std: float = 0.05  # fraction of per-channel std

    apply_temporal_crop: bool = True
    temporal_crop_ratio: float = 0.1  # max fraction of time-points to drop from each end

    apply_channel_dropout: bool = True
    channel_dropout_prob: float = 0.1  # probability of zeroing each channel

    apply_amplitude_scale: bool = True
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)

    # Spectrogram-level augmentation (SpecAugment, applied after CWT)
    apply_freq_mask: bool = True
    freq_mask_max_width: int = 8  # max frequency bins to mask

    apply_time_mask: bool = True
    time_mask_max_width: int = 16  # max time-step columns to mask


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    # Image branch backbone (ViT-Tiny)
    vit_model_name: str = "vit_tiny_patch16_224"
    vit_pretrained: bool = True
    vit_drop_rate: float = 0.1
    # Number of input channels for the ViT (3 for RGB legacy, 9 for multichannel)
    in_chans: int = 3

    # Math branch
    csp_n_components: int = 6
    math_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    math_drop_rate: float = 0.3

    # Fusion
    fusion_method: str = "attention"  # "attention", "gated"
    fused_dim: int = 128

    # Classifier head
    classifier_hidden_dim: int = 64
    n_classes: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    warmup_epochs: int = 5

    # Regularization
    label_smoothing: float = 0.1

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Evaluation
    eval_strategy: str = "cross_subject"  # "within_subject", "cross_subject", "cross_dataset"
    n_folds: int = 5

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 4

    # Logging
    log_dir: str = str(OUTPUTS_DIR / "runs")
    checkpoint_dir: str = str(CHECKPOINTS_DIR)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default"
    description: str = ""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass from a (possibly nested) dictionary."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key in field_types and isinstance(value, dict):
            # Resolve the field type for nested dataclasses
            ft = field_types[key]
            # Handle string annotations
            if isinstance(ft, str):
                ft = eval(ft)  # noqa: S307 - safe here, types are our own dataclasses
            if dataclasses.is_dataclass(ft):
                kwargs[key] = _dataclass_from_dict(ft, value)
            else:
                kwargs[key] = value
        elif key in field_types:
            kwargs[key] = value
        else:
            logger.warning("Unknown config key: %s", key)
    return cls(**kwargs)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load experiment configuration from a YAML file with optional overrides.

    Args:
        config_path: Path to a YAML config file. If None, returns defaults.
        overrides: Dictionary of overrides to apply on top of the config file.

    Returns:
        A fully populated ExperimentConfig.
    """
    config_dict: dict[str, Any] = {}

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", config_path)

    if overrides:
        config_dict = _merge_dict(config_dict, overrides)

    return _dataclass_from_dict(ExperimentConfig, config_dict)
