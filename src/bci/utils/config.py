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
            "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
        ]
    )


@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing."""

    # Filtering
    l_freq: float = 4.0
    h_freq: float = 40.0
    notch_freq: float | None = 50.0

    # ICA
    apply_ica: bool = True
    n_ica_components: int | None = None  # None = use all channels
    ica_method: str = "fastica"

    # Epoching
    tmin: float = 0.0
    tmax: float = 4.0

    # Resampling
    resample_freq: float | None = 128.0

    # Normalization
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
    # "rgb_c3_cz_c4" = map C3→R, Cz→G, C4→B
    # "mosaic" = tile multiple channel spectrograms
    channel_mode: str = "rgb_c3_cz_c4"
    output_dir: str = str(DATA_DIR / "spectrograms")


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    # Image branch (EfficientNet-B0 by default; ViT also supported)
    vit_model_name: str = "efficientnet_b0"
    vit_pretrained: bool = True
    vit_drop_rate: float = 0.1

    # Math branch
    csp_n_components: int = 6
    math_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    math_drop_rate: float = 0.3

    # Fusion
    fusion_method: str = "attention"  # "attention", "concat", "gated"
    fused_dim: int = 256

    # Classifier head
    classifier_hidden_dim: int = 128
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
