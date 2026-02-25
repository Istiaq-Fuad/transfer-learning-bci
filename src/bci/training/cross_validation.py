"""Cross-validation strategies for MI-EEG classification.

Supports three evaluation paradigms:
    1. Within-subject k-fold: standard k-fold CV per subject
    2. Cross-subject LOSO: leave-one-subject-out across all subjects
    3. Cross-dataset: train on source dataset, test on target dataset

All strategies return per-fold results and aggregate statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single fold."""
    fold: int
    subject: int | None
    accuracy: float
    kappa: float
    f1_macro: float
    n_train: int
    n_test: int
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    y_prob: np.ndarray | None = field(default=None, repr=False)


@dataclass
class CVResult:
    """Aggregated cross-validation results."""
    strategy: str
    model_name: str
    folds: list[FoldResult]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([f.accuracy for f in self.folds]))

    @property
    def std_accuracy(self) -> float:
        return float(np.std([f.accuracy for f in self.folds]))

    @property
    def mean_kappa(self) -> float:
        return float(np.mean([f.kappa for f in self.folds]))

    @property
    def mean_f1(self) -> float:
        return float(np.mean([f.f1_macro for f in self.folds]))

    @property
    def per_subject_accuracy(self) -> dict[int, float]:
        """Average accuracy grouped by subject (for LOSO)."""
        from collections import defaultdict
        acc_by_subject: dict[int, list[float]] = defaultdict(list)
        for fold in self.folds:
            if fold.subject is not None:
                acc_by_subject[fold.subject].append(fold.accuracy)
        return {s: float(np.mean(accs)) for s, accs in acc_by_subject.items()}

    def summary(self) -> str:
        lines = [
            f"Model : {self.model_name}",
            f"Strategy: {self.strategy}",
            f"Folds : {len(self.folds)}",
            f"Accuracy: {self.mean_accuracy:.2f}% ± {self.std_accuracy:.2f}%",
            f"Kappa   : {self.mean_kappa:.4f}",
            f"F1 macro: {self.mean_f1:.4f}",
        ]
        if self.strategy == "loso":
            lines.append("Per-subject:")
            for subj, acc in sorted(self.per_subject_accuracy.items()):
                lines.append(f"  S{subj:02d}: {acc:.2f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Type alias: a fit+predict callable that takes (X_train, y_train, X_test)
#             and returns (y_pred, y_prob_or_None)
# ---------------------------------------------------------------------------
PredictFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray | None],
]


def _compute_fold_metrics(
    fold: int,
    subject: int | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_train: int,
    n_test: int,
) -> FoldResult:
    from bci.training.evaluation import compute_metrics
    m = compute_metrics(y_true, y_pred, y_prob)
    return FoldResult(
        fold=fold,
        subject=subject,
        accuracy=m["accuracy"],
        kappa=m["kappa"],
        f1_macro=m["f1_macro"],
        n_train=n_train,
        n_test=n_test,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: PredictFn,
    model_name: str = "model",
    n_folds: int = 5,
    subject_id: int = 0,
    seed: int = 42,
) -> CVResult:
    """Stratified k-fold cross-validation for a single subject.

    Args:
        X: EEG data (n_trials, n_channels, n_times) or features (n_trials, n_features).
        y: Integer labels (n_trials,).
        predict_fn: Callable(X_train, y_train, X_test) -> (y_pred, y_prob).
        model_name: Name for reporting.
        n_folds: Number of CV folds.
        subject_id: Subject identifier for reporting.
        seed: Random seed for fold splitting.

    Returns:
        CVResult with per-fold and aggregate metrics.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        y_pred, y_prob = predict_fn(X_train, y_train, X_test)

        fold = _compute_fold_metrics(
            fold=fold_idx,
            subject=subject_id,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            n_train=len(train_idx),
            n_test=len(test_idx),
        )
        folds.append(fold)
        logger.info(
            "  [S%02d fold %d/%d] acc=%.2f%% kappa=%.3f",
            subject_id, fold_idx + 1, n_folds, fold.accuracy, fold.kappa,
        )

    result = CVResult(strategy="within_subject", model_name=model_name, folds=folds)
    logger.info("  Subject %02d mean: %.2f%% ± %.2f%%", subject_id, result.mean_accuracy, result.std_accuracy)
    return result


def within_subject_cv_all(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    predict_fn: PredictFn,
    model_name: str = "model",
    n_folds: int = 5,
    seed: int = 42,
) -> CVResult:
    """Run within-subject CV for all subjects and pool results.

    Args:
        subject_data: Dict of subject_id -> (X, y).
        predict_fn: Callable(X_train, y_train, X_test) -> (y_pred, y_prob).
        model_name: Name for reporting.
        n_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        CVResult pooling all subjects' folds.
    """
    all_folds: list[FoldResult] = []
    for subj_id, (X, y) in sorted(subject_data.items()):
        logger.info("Within-subject CV: subject %d (%d trials)...", subj_id, len(y))
        result = within_subject_cv(X, y, predict_fn, model_name, n_folds, subj_id, seed)
        all_folds.extend(result.folds)

    return CVResult(strategy="within_subject", model_name=model_name, folds=all_folds)


def loso_cv(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    predict_fn: PredictFn,
    model_name: str = "model",
) -> CVResult:
    """Leave-One-Subject-Out cross-validation.

    Each fold: train on all subjects except one, test on the left-out subject.

    Args:
        subject_data: Dict of subject_id -> (X, y).
        predict_fn: Callable(X_train, y_train, X_test) -> (y_pred, y_prob).
        model_name: Name for reporting.

    Returns:
        CVResult with one fold per subject.
    """
    subjects = sorted(subject_data.keys())
    folds: list[FoldResult] = []

    for fold_idx, test_subj in enumerate(subjects):
        # Build train set from all other subjects
        train_Xs = [subject_data[s][0] for s in subjects if s != test_subj]
        train_ys = [subject_data[s][1] for s in subjects if s != test_subj]
        X_train = np.concatenate(train_Xs, axis=0)
        y_train = np.concatenate(train_ys, axis=0)

        X_test, y_test = subject_data[test_subj]

        logger.info(
            "LOSO fold %d/%d: test=S%02d, train=%d trials, test=%d trials",
            fold_idx + 1, len(subjects), test_subj, len(y_train), len(y_test),
        )

        y_pred, y_prob = predict_fn(X_train, y_train, X_test)

        fold = _compute_fold_metrics(
            fold=fold_idx,
            subject=test_subj,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            n_train=len(y_train),
            n_test=len(y_test),
        )
        folds.append(fold)
        logger.info(
            "  -> S%02d: acc=%.2f%% kappa=%.3f",
            test_subj, fold.accuracy, fold.kappa,
        )

    return CVResult(strategy="loso", model_name=model_name, folds=folds)


def make_synthetic_subject_data(
    n_subjects: int = 9,
    n_trials_per_subject: int = 144,
    n_channels: int = 22,
    n_times: int = 512,
    seed: int = 42,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic subject data for testing pipelines.

    Produces data with a slight signal difference between classes
    (mu/beta band power) so classifiers can do better than chance.

    Args:
        n_subjects: Number of subjects.
        n_trials_per_subject: Trials per subject (half per class).
        n_channels: Number of EEG channels.
        n_times: Time points per trial.
        seed: Random seed.

    Returns:
        Dict of subject_id -> (X, y) arrays.
    """
    rng = np.random.default_rng(seed)
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for subj in range(1, n_subjects + 1):
        n_per_class = n_trials_per_subject // 2
        X_list = []
        y_list = []

        for cls in range(2):
            X_cls = rng.standard_normal((n_per_class, n_channels, n_times)).astype(np.float32)
            # Add a class-discriminative signal on channels 3 (C3) and 4 (C4)
            # Class 0 (left): higher power on C4 (right hemisphere)
            # Class 1 (right): higher power on C3 (left hemisphere)
            t = np.linspace(0, 4, n_times)
            signal = np.sin(2 * np.pi * 10 * t) * 2.0  # 10 Hz mu rhythm
            if cls == 0:
                X_cls[:, 3, :] += signal
            else:
                X_cls[:, 4, :] += signal

            X_list.append(X_cls)
            y_list.append(np.full(n_per_class, cls, dtype=np.int64))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # Shuffle
        idx = rng.permutation(len(y))
        subject_data[subj] = (X[idx], y[idx])

    return subject_data
