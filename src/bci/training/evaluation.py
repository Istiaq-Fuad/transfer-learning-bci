"""Evaluation metrics and utilities for MI-EEG classification."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard MI-EEG classification metrics.

    Args:
        y_true: True labels of shape (n_samples,).
        y_pred: Predicted labels of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples, n_classes).
            Required for AUC-ROC.

    Returns:
        Dictionary with metric names and values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    # AUC-ROC (requires probability estimates)
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                # Binary classification: use probability of positive class
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc_roc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr",
                )
        except Exception:
            logger.warning("Could not compute AUC-ROC")

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """Print and return a formatted classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names for the classes.

    Returns:
        Formatted classification report string.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\n%s", report)
    return report
