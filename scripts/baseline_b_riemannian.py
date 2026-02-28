"""Baseline B: Riemannian geometry + LDA classification pipeline.

Runs within-subject 5-fold CV and LOSO CV using:
    - Riemannian tangent space features (covariance estimation + TangentSpace)
    - Configurable classifier: LDA (with shrinkage), SVM, MDM, or FgMDM

Supports two methods via --method:
    basic  – full-band tangent space (original baseline)
    fbts   – Filter Bank Tangent Space (per-band covariance + TangentSpace)

Supports Riemannian Alignment (RA) for cross-subject generalisation via
--align flag (particularly effective for LOSO).

Usage:
    # Synthetic data (no download required)
    uv run python scripts/baseline_b_riemannian.py

    # Real data — FBTS + MDM classifier
    uv run python scripts/baseline_b_riemannian.py --data real --method fbts --classifier mdm

    # Real data — basic + Riemannian alignment + SVM
    uv run python scripts/baseline_b_riemannian.py --data real --align --classifier svm
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from bci.features.riemannian import (
    FBRiemannianFeatureExtractor,
    RiemannianFeatureExtractor,
    riemannian_recenter,
)
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


def _make_classifier(classifier: str, sfreq: float):
    """Instantiate the requested classifier."""
    if classifier == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if classifier == "svm":
        from sklearn.svm import SVC

        return SVC(kernel="rbf", probability=True, C=1.0, gamma="scale")
    if classifier == "mdm":
        from pyriemann.classification import MDM

        return MDM(metric="riemann")
    if classifier == "fgmdm":
        from pyriemann.classification import FgMDM

        return FgMDM(metric="riemann", tsupdate=False)
    raise ValueError(f"Unknown classifier: {classifier!r}. Choose lda | svm | mdm | fgmdm")


def make_predict_fn(
    method: str = "basic",
    estimator: str = "oas",
    metric: str = "riemann",
    classifier: str = "lda",
    sfreq: float = 128.0,
    align: bool = False,
):
    """Return a predict_fn for Riemannian + classifier.

    Args:
        method: "basic" | "fbts" (filter bank tangent space).
        estimator: Covariance estimator — "oas" | "lwf" | "scm".
        metric: Riemannian metric — "riemann" | "logeuclid" | "euclid".
        classifier: "lda" | "svm" | "mdm" | "fgmdm".
        sfreq: Sampling frequency in Hz.
        align: If True, apply Riemannian recentering before feature extraction.

    Note: MDM and FgMDM work directly on covariance matrices and bypass
    tangent space projection. They use pyriemann's own internal pipeline.
    """
    use_mdm = classifier in ("mdm", "fgmdm")

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_tr, X_te = X_train, X_test

        if align:
            X_tr, X_te = riemannian_recenter(X_tr, X_te, estimator=estimator)

        if use_mdm:
            # MDM/FgMDM classify directly from covariance matrices
            from pyriemann.estimation import Covariances

            cov_est = Covariances(estimator=estimator)
            covs_train = cov_est.fit_transform(X_tr.astype(np.float64))
            covs_test = cov_est.transform(X_te.astype(np.float64))

            clf = _make_classifier(classifier, sfreq)
            clf.fit(covs_train, y_train)
            y_pred = clf.predict(covs_test)
            y_prob = clf.predict_proba(covs_test)
            return y_pred, y_prob

        # Tangent-space path (basic or fbts)
        if method == "fbts":
            extractor: FBRiemannianFeatureExtractor | RiemannianFeatureExtractor = (
                FBRiemannianFeatureExtractor(
                    estimator=estimator,
                    metric=metric,
                    sfreq=sfreq,
                )
            )
        else:
            extractor = RiemannianFeatureExtractor(estimator=estimator, metric=metric)

        clf = _make_classifier(classifier, sfreq)

        features_train = extractor.fit_transform(X_tr, y_train)
        features_test = extractor.transform(X_te)

        clf.fit(features_train, y_train)
        y_pred = clf.predict(features_test)
        y_prob = clf.predict_proba(features_test)

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
    paradigm = LeftRightImagery(fmin=8.0, fmax=32.0, resample=128.0)

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
        "--method",
        type=str,
        default="basic",
        choices=["basic", "fbts"],
        help="Feature extraction method: basic (full-band TS) or fbts (filter bank TS). Default: basic",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="oas",
        choices=["lwf", "scm", "oas"],
        help="Covariance estimator (default: oas)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="riemann",
        choices=["riemann", "logeuclid", "euclid"],
        help="Riemannian metric (default: riemann)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="lda",
        choices=["lda", "svm", "mdm", "fgmdm"],
        help="Classifier to use (default: lda)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        default=False,
        help="Apply Riemannian Alignment before feature extraction",
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
    logger.info("  Method:     %s", args.method)
    logger.info("  Estimator:  %s", args.estimator)
    logger.info("  Metric:     %s", args.metric)
    logger.info("  Classifier: %s", args.classifier)
    logger.info("  Align:      %s", args.align)
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

    predict_fn = make_predict_fn(
        method=args.method,
        estimator=args.estimator,
        metric=args.metric,
        classifier=args.classifier,
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
