"""Training and evaluation utilities."""

from bci.training.cross_validation import (
    CVResult,
    FoldResult,
    PredictFn,
    loso_cv,
    make_synthetic_subject_data,
    within_subject_cv,
    within_subject_cv_all,
)
from bci.training.evaluation import compute_metrics
from bci.training.trainer import EpochResult, TrainResult, Trainer

__all__ = [
    # cross_validation
    "CVResult",
    "FoldResult",
    "PredictFn",
    "loso_cv",
    "make_synthetic_subject_data",
    "within_subject_cv",
    "within_subject_cv_all",
    # evaluation
    "compute_metrics",
    # trainer
    "EpochResult",
    "TrainResult",
    "Trainer",
]
