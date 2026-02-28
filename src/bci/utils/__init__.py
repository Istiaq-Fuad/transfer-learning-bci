"""Utility functions for visualization, logging, and configuration."""

from bci.utils.config import (
    DEFAULT_CLS_HIDDEN,
    DEFAULT_FUSED_DIM,
    FILTER_BANK_BANDS,
    VIT_FEATURE_DIM,
    AugmentationConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessingConfig,
    SpectrogramConfig,
    TrainingConfig,
    load_config,
)
from bci.utils.logging import setup_stage_logging
from bci.utils.seed import set_seed
from bci.utils.visualization import save_fold_plots

__all__ = [
    # config constants
    "DEFAULT_CLS_HIDDEN",
    "DEFAULT_FUSED_DIM",
    "FILTER_BANK_BANDS",
    "VIT_FEATURE_DIM",
    # config dataclasses
    "AugmentationConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "SpectrogramConfig",
    "TrainingConfig",
    "load_config",
    # logging
    "setup_stage_logging",
    # seed
    "set_seed",
    # visualization
    "save_fold_plots",
]
