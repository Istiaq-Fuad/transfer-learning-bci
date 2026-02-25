"""Baseline B: Riemannian geometry + LDA classification pipeline.

Runs within-subject 5-fold CV and LOSO CV using:
    - Riemannian tangent space features (Ledoit-Wolf covariance, Riemannian metric)
    - Linear Discriminant Analysis (LDA) classifier

The Riemannian approach treats EEG covariance matrices as points on an SPD
manifold and projects them into a Euclidean tangent space for classification.

Usage:
    # Synthetic data (no download required)
    uv run python scripts/baseline_b_riemannian.py

    # Real data (requires BCI IV-2a downloaded)
    uv run python scripts/baseline_b_riemannian.py --data real --data-dir ~/mne_data
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from bci.features.riemannian import RiemannianFeatureExtractor
from bci.training.cross_validation import (
    CVResult,
    loso_cv,
    make_synthetic_subject_data,
    within_subject_cv_all,
)
from bci.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("mne").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

MODEL_NAME = "Riemannian+LDA"


def make_predict_fn(estimator: str = "lwf", metric: str = "riemann"):
    """Return a predict_fn for Riemannian tangent space + LDA.

    Each fold gets a fresh pipeline to avoid data leakage.
    """
    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        riemann = RiemannianFeatureExtractor(estimator=estimator, metric=metric)
        lda = LinearDiscriminantAnalysis()

        features_train = riemann.fit_transform(X_train, y_train)
        features_test = riemann.transform(X_test)

        lda.fit(features_train, y_train)
        y_pred = lda.predict(features_test)
        y_prob = lda.predict_proba(features_test)

        return y_pred, y_prob

    return predict_fn


def print_results_table(within_result: CVResult, loso_result: CVResult) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 60)
    print(f"  BASELINE B: {MODEL_NAME}")
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
    """Load BCI IV-2a data using MOABB + MNE preprocessing."""
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")

    logger.info("Loading BCI IV-2a dataset from %s ...", data_dir)
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery()

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subject_id in dataset.subject_list:
        try:
            X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[subject_id])
            classes = sorted(np.unique(y_labels))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[lbl] for lbl in y_labels], dtype=np.int64)
            subject_data[subject_id] = (X.astype(np.float32), y)
            logger.info("  Subject %d: X=%s", subject_id, X.shape)
        except Exception as e:
            logger.warning("  Skipping subject %d: %s", subject_id, e)

    return subject_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline B: Riemannian + LDA")
    parser.add_argument(
        "--data",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    parser.add_argument("--data-dir", type=str, default="~/mne_data", help="MNE data dir")
    parser.add_argument(
        "--estimator",
        type=str,
        default="lwf",
        choices=["lwf", "scm", "oas"],
        help="Covariance estimator (default: lwf)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="riemann",
        choices=["riemann", "logeuclid", "euclid"],
        help="Riemannian metric (default: riemann)",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--n-subjects", type=int, default=9, help="Subjects (synthetic only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_b_riemannian.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info("=" * 50)
    logger.info("Baseline B: Riemannian + LDA")
    logger.info("  Data:       %s", args.data)
    logger.info("  Estimator:  %s", args.estimator)
    logger.info("  Metric:     %s", args.metric)
    logger.info("  CV folds:   %d", args.n_folds)
    logger.info("  Seed:       %d", args.seed)
    logger.info("=" * 50)

    # --- Load data ---
    if args.data == "synthetic":
        logger.info("Generating synthetic data (%d subjects)...", args.n_subjects)
        subject_data = make_synthetic_subject_data(n_subjects=args.n_subjects)
    else:
        subject_data = load_real_data(args.data_dir)

    if not subject_data:
        logger.error("No data loaded. Exiting.")
        return

    predict_fn = make_predict_fn(estimator=args.estimator, metric=args.metric)

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
