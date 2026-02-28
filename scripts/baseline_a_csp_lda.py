"""Baseline A: CSP + LDA classification pipeline.

Runs within-subject 5-fold CV and LOSO CV using:
    - Common Spatial Patterns (CSP, Ledoit-Wolf regularization, 6 components)
    - Linear Discriminant Analysis (LDA) with Ledoit-Wolf shrinkage

Supports three methods via --method:
    basic    – single-band CSP + LDA (original baseline)
    fbcsp    – Filter Bank CSP across 6 sub-bands (8–32 Hz) + LDA
    ensemble – multi-window FBCSP ensemble (majority vote)

Supports cross-subject covariance alignment via --align flag (LOSO only).

Supports both synthetic data (default) and real BCI IV-2a data.

Usage:
    # Synthetic data (no download required)
    uv run python scripts/baseline_a_csp_lda.py

    # Real data — basic CSP
    uv run python scripts/baseline_a_csp_lda.py --data real --data-dir ~/mne_data

    # Real data — FBCSP
    uv run python scripts/baseline_a_csp_lda.py --data real --method fbcsp

    # Real data — ensemble + Riemannian alignment for LOSO
    uv run python scripts/baseline_a_csp_lda.py --data real --method ensemble --align
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from bci.features.csp import (
    CSPFeatureExtractor,
    EnsembleCSPClassifier,
    FBCSPFeatureExtractor,
)
from bci.training.cross_validation import (
    CVResult,
    loso_cv,
    make_synthetic_subject_data,
    within_subject_cv_all,
)
from bci.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Silence noisy MNE output (CSP fitting messages)
logging.getLogger("mne").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

MODEL_NAME = "CSP+LDA"


def make_predict_fn(
    method: str = "basic",
    n_components: int = 6,
    sfreq: float = 128.0,
    align: bool = False,
):
    """Return a predict_fn for CSP + LDA (or FBCSP, or ensemble).

    Args:
        method: "basic" | "fbcsp" | "ensemble"
        n_components: CSP components (used for basic and fbcsp).
        sfreq: Sampling frequency — needed for FBCSP band filtering.
        align: If True, apply subject-specific Riemannian covariance alignment
            before CSP (helps LOSO). Requires pyriemann.

    Each fold gets a fresh pipeline to avoid data leakage.
    """

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_tr, X_te = X_train, X_test

        if align:
            X_tr, X_te = _riemannian_align(X_tr, X_te)

        if method == "ensemble":
            clf = EnsembleCSPClassifier(sfreq=sfreq)
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)
            return y_pred, y_prob

        if method == "fbcsp":
            extractor: FBCSPFeatureExtractor | CSPFeatureExtractor = FBCSPFeatureExtractor(
                sfreq=sfreq, n_components=4, k_best=12
            )
        else:  # basic
            extractor = CSPFeatureExtractor(n_components=n_components, reg="ledoit_wolf")

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        features_train = extractor.fit_transform(X_tr, y_train)
        features_test = extractor.transform(X_te)

        lda.fit(features_train, y_train)
        y_pred = lda.predict(features_test)
        y_prob = lda.predict_proba(features_test)

        return y_pred, y_prob

    return predict_fn


def _riemannian_align(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Riemannian recentering: project both train and test to the identity.

    Computes the Riemannian mean of training covariance matrices, then maps
    both sets to the tangent point at the identity matrix using the
    whitening transform  W = mean_cov^{-1/2}.

    Args:
        X_train: (n_train, n_channels, n_times)
        X_test:  (n_test,  n_channels, n_times)

    Returns:
        Aligned X_train and X_test with the same shapes.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    from scipy.linalg import inv, sqrtm

    cov_est = Covariances(estimator="oas")
    covs_train = cov_est.fit_transform(X_train.astype(np.float64))

    ref = mean_riemann(covs_train)  # (n_channels, n_channels)
    W = np.real(inv(sqrtm(ref)))  # whitening matrix

    def _apply_whitening(X: np.ndarray) -> np.ndarray:
        # X: (n_trials, n_channels, n_times)
        # apply spatial filter W: X_aligned[i] = W @ X[i]
        return np.einsum("ij,bjt->bit", W, X.astype(np.float64)).astype(np.float32)

    return _apply_whitening(X_train), _apply_whitening(X_test)


def print_results_table(within_result: CVResult, loso_result: CVResult) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 60)
    print(f"  BASELINE A: {MODEL_NAME}")
    print("=" * 60)
    print(f"\n{'Strategy':<25} {'Accuracy':>10} {'±':>4} {'Std':>8} {'Kappa':>8} {'F1':>8}")
    print("-" * 60)
    print(
        f"{'Within-Subject (5-fold)':<25} "
        f"{within_result.mean_accuracy:>9.2f}% "
        f"{'':>4} "
        f"{within_result.std_accuracy:>7.2f}% "
        f"{within_result.mean_kappa:>8.4f} "
        f"{within_result.mean_f1:>8.4f}"
    )
    print(
        f"{'LOSO (cross-subject)':<25} "
        f"{loso_result.mean_accuracy:>9.2f}% "
        f"{'':>4} "
        f"{loso_result.std_accuracy:>7.2f}% "
        f"{loso_result.mean_kappa:>8.4f} "
        f"{loso_result.mean_f1:>8.4f}"
    )
    print("=" * 60)

    # Per-subject breakdown for LOSO
    if loso_result.per_subject_accuracy:
        print("\nLOSO per-subject accuracy:")
        for subj, acc in sorted(loso_result.per_subject_accuracy.items()):
            bar = "#" * int(acc / 5)
            print(f"  S{subj:02d}: {acc:6.2f}%  {bar}")
    print()


def save_results(
    within_result: CVResult,
    loso_result: CVResult,
    output_path: Path,
) -> None:
    """Save results to a JSON file."""
    results = {
        "model": MODEL_NAME,
        "within_subject": {
            "mean_accuracy": within_result.mean_accuracy,
            "std_accuracy": within_result.std_accuracy,
            "mean_kappa": within_result.mean_kappa,
            "mean_f1": within_result.mean_f1,
            "n_folds": len(within_result.folds),
        },
        "loso": {
            "mean_accuracy": loso_result.mean_accuracy,
            "std_accuracy": loso_result.std_accuracy,
            "mean_kappa": loso_result.mean_kappa,
            "mean_f1": loso_result.mean_f1,
            "n_folds": len(loso_result.folds),
            "per_subject": loso_result.per_subject_accuracy,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


def load_real_data(data_dir: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load BCI IV-2a data using MOABB + MNE preprocessing.

    Args:
        data_dir: Path to MNE data directory.

    Returns:
        Dict of subject_id -> (X, y) where X is (n_trials, n_channels, n_times).
    """
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")

    logger.info("Loading BCI IV-2a dataset from %s ...", data_dir)
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8.0, fmax=32.0, resample=128.0)

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for subject_id in dataset.subject_list:
        try:
            X, y_labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
            # Convert string labels to integers
            classes = sorted(np.unique(y_labels))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[lbl] for lbl in y_labels], dtype=np.int64)
            subject_data[subject_id] = (X.astype(np.float32), y)
            logger.info("  Subject %d: X=%s, classes=%s", subject_id, X.shape, classes)
        except Exception as e:
            logger.warning("  Skipping subject %d: %s", subject_id, e)

    return subject_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline A: CSP + LDA")
    parser.add_argument(
        "--data",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    parser.add_argument("--data-dir", type=str, default="~/mne_data", help="MNE data dir")
    parser.add_argument("--n-components", type=int, default=6, help="CSP components (basic mode)")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--n-subjects", type=int, default=9, help="Subjects (synthetic only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--method",
        choices=["basic", "fbcsp", "ensemble"],
        default="basic",
        help="CSP variant: basic | fbcsp (filter bank) | ensemble (multi-window, default: basic)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply Riemannian covariance alignment before CSP (helps LOSO)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_a_csp_lda.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info("=" * 50)
    logger.info("Baseline A: CSP + LDA")
    logger.info("  Data:         %s", args.data)
    logger.info("  Method:       %s", args.method)
    logger.info("  Align:        %s", args.align)
    logger.info("  CSP components: %d", args.n_components)
    logger.info("  CV folds:     %d", args.n_folds)
    logger.info("  Seed:         %d", args.seed)
    logger.info("=" * 50)

    # Sampling frequency: 128 Hz for real data (MOABB resample=128); synthetic uses 128 too.
    sfreq = 128.0

    # --- Load data ---
    if args.data == "synthetic":
        logger.info("Generating synthetic data (%d subjects)...", args.n_subjects)
        subject_data = make_synthetic_subject_data(n_subjects=args.n_subjects)
    else:
        subject_data = load_real_data(args.data_dir)

    if not subject_data:
        logger.error("No data loaded. Exiting.")
        return

    predict_fn = make_predict_fn(
        method=args.method,
        n_components=args.n_components,
        sfreq=sfreq,
        align=args.align,
    )

    # --- Within-subject CV ---
    logger.info("\nRunning within-subject %d-fold CV...", args.n_folds)
    t0 = time.time()
    within_result = within_subject_cv_all(
        subject_data,
        predict_fn,
        model_name=MODEL_NAME,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    t_within = time.time() - t0
    logger.info("Within-subject CV done in %.1fs", t_within)

    # --- LOSO CV ---
    logger.info("\nRunning LOSO CV...")
    t0 = time.time()
    loso_result = loso_cv(subject_data, predict_fn, model_name=MODEL_NAME)
    t_loso = time.time() - t0
    logger.info("LOSO CV done in %.1fs", t_loso)

    # --- Print and save ---
    print_results_table(within_result, loso_result)
    save_results(within_result, loso_result, Path(args.output))


if __name__ == "__main__":
    main()
