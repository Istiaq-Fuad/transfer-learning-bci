"""Improved baselines: MDM, TS+SVM, TS+LogReg, FBCSP+SVM, CSP+SVM, Ensemble.

Runs within-subject 5-fold CV and LOSO CV with improved classification
pipelines identified through literature review:

  1. MDM (Minimum Distance to Mean) — Riemannian classifier, no tangent space
  2. TS + SVM — Tangent space features + Support Vector Machine
  3. TS + LogReg — Tangent space features + Logistic Regression (L2)
  4. FBCSP + SVM — Filter Bank CSP with MIBIF feature selection + SVM
  5. CSP + SVM — Standard CSP features + SVM (improved over LDA baseline)
  6. Ensemble — Majority vote of MDM + TS+SVM + CSP+SVM

Usage:
    # Synthetic data (no download required)
    uv run python scripts/baseline_improved.py

    # Real data (requires BCI IV-2a downloaded)
    uv run python scripts/baseline_improved.py --data real --data-dir ~/mne_data

    # Run a specific pipeline only
    uv run python scripts/baseline_improved.py --data real --pipelines mdm ts_svm fbcsp_svm
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from bci.features.csp import CSPFeatureExtractor
from bci.features.fbcsp import FBCSPFeatureExtractor
from bci.training.cross_validation import (
    CVResult,
    PredictFn,
    loso_cv,
    make_synthetic_subject_data,
    within_subject_cv_all,
)
from bci.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("mne").setLevel(logging.ERROR)
logging.getLogger("bci").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline factories: each returns a PredictFn
# ---------------------------------------------------------------------------


def make_mdm_predict_fn(
    estimator: str = "lwf",
    metric: str = "riemann",
) -> PredictFn:
    """MDM: Minimum Distance to Mean on SPD manifold.

    This is the simplest Riemannian classifier — no tangent space projection,
    no secondary classifier. Classification is based purely on geodesic
    distance to class means on the SPD manifold.

    Expected improvement over Riemann+LDA: +8-18% (fixes the dimensionality
    mismatch problem where LDA fails on 253-dim tangent space with ~72 samples).
    """

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        pipe = Pipeline(
            [
                ("covariances", Covariances(estimator=estimator)),
                ("mdm", MDM(metric=metric)),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # MDM can return distance-based probabilities
        try:
            y_prob = pipe.predict_proba(X_test)
        except Exception:
            y_prob = None

        return y_pred, y_prob

    return predict_fn


def make_ts_svm_predict_fn(
    estimator: str = "lwf",
    metric: str = "riemann",
    C: float = 1.0,
    kernel: str = "rbf",
) -> PredictFn:
    """Tangent space + SVM.

    Projects covariance matrices to tangent space (Euclidean), then classifies
    with SVM. SVM handles the high dimensionality (253-dim) much better than
    LDA when sample size is small.

    Expected improvement over Riemann+LDA: +3-8%.
    """

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        pipe = Pipeline(
            [
                ("covariances", Covariances(estimator=estimator)),
                ("tangent_space", TangentSpace(metric=metric)),
                ("svm", SVC(C=C, kernel=kernel, probability=True, random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)
        return y_pred, y_prob

    return predict_fn


def make_ts_logreg_predict_fn(
    estimator: str = "lwf",
    metric: str = "riemann",
    C: float = 1.0,
) -> PredictFn:
    """Tangent space + Logistic Regression (L2 regularized).

    Similar to TS+SVM but with logistic regression, which is faster and
    provides well-calibrated probabilities. L2 regularization handles the
    curse of dimensionality better than LDA.

    Expected improvement over Riemann+LDA: +3-8%.
    """

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        pipe = Pipeline(
            [
                ("covariances", Covariances(estimator=estimator)),
                ("tangent_space", TangentSpace(metric=metric)),
                ("logreg", LogisticRegression(C=C, max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)
        return y_pred, y_prob

    return predict_fn


def make_fbcsp_svm_predict_fn(
    n_components: int = 4,
    n_features_select: int = 12,
    sfreq: float = 128.0,
    C: float = 1.0,
) -> PredictFn:
    """Filter Bank CSP + SVM.

    Decomposes signal into 9 sub-bands (4 Hz each, 4-40 Hz), applies CSP
    per band, selects top features via mutual information, classifies with SVM.

    Expected improvement over CSP+LDA: +5-12%.
    """

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        fbcsp = FBCSPFeatureExtractor(
            n_components=n_components,
            freq_min=4.0,
            freq_max=40.0,
            band_width=4.0,
            sfreq=sfreq,
            n_features_select=n_features_select,
        )
        svm = SVC(C=C, kernel="rbf", probability=True, random_state=42)

        features_train = fbcsp.fit_transform(X_train, y_train)
        features_test = fbcsp.transform(X_test)

        svm.fit(features_train, y_train)
        y_pred = svm.predict(features_test)
        y_prob = svm.predict_proba(features_test)
        return y_pred, y_prob

    return predict_fn


def make_csp_svm_predict_fn(
    n_components: int = 6,
    C: float = 1.0,
) -> PredictFn:
    """CSP + SVM (improved over CSP + LDA baseline).

    Same CSP feature extraction as Baseline A, but with SVM classifier
    instead of LDA. SVM is more robust for small sample sizes.

    Expected improvement over CSP+LDA: +2-5%.
    """

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        csp = CSPFeatureExtractor(n_components=n_components, reg="ledoit_wolf")
        svm = SVC(C=C, kernel="rbf", probability=True, random_state=42)

        features_train = csp.fit_transform(X_train, y_train)
        features_test = csp.transform(X_test)

        svm.fit(features_train, y_train)
        y_pred = svm.predict(features_test)
        y_prob = svm.predict_proba(features_test)
        return y_pred, y_prob

    return predict_fn


def make_ensemble_predict_fn(
    n_csp_components: int = 6,
) -> PredictFn:
    """Ensemble: majority vote of MDM + TS+SVM + CSP+SVM.

    Combines three complementary classifiers:
    - MDM: Riemannian manifold classification (no projection artifacts)
    - TS+SVM: Tangent space with powerful nonlinear classifier
    - CSP+SVM: Spatial filter-based features with SVM

    Expected improvement over best single pipeline: +2-5%.
    """

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        n_test = X_test.shape[0]

        # --- MDM ---
        mdm_pipe = Pipeline(
            [
                ("covariances", Covariances(estimator="lwf")),
                ("mdm", MDM(metric="riemann")),
            ]
        )
        mdm_pipe.fit(X_train, y_train)
        y_pred_mdm = mdm_pipe.predict(X_test)

        # --- TS + SVM ---
        ts_svm_pipe = Pipeline(
            [
                ("covariances", Covariances(estimator="lwf")),
                ("tangent_space", TangentSpace(metric="riemann")),
                ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        )
        ts_svm_pipe.fit(X_train, y_train)
        y_pred_ts = ts_svm_pipe.predict(X_test)
        y_prob_ts = ts_svm_pipe.predict_proba(X_test)

        # --- CSP + SVM ---
        csp = CSPFeatureExtractor(n_components=n_csp_components, reg="ledoit_wolf")
        svm = SVC(kernel="rbf", probability=True, random_state=42)
        feats_train = csp.fit_transform(X_train, y_train)
        feats_test = csp.transform(X_test)
        svm.fit(feats_train, y_train)
        y_pred_csp = svm.predict(feats_test)

        # --- Majority vote ---
        votes = np.stack([y_pred_mdm, y_pred_ts, y_pred_csp], axis=0)  # (3, n_test)
        y_pred = np.array(
            [np.bincount(votes[:, i]).argmax() for i in range(n_test)],
            dtype=np.int64,
        )

        # Use TS+SVM probabilities as ensemble probability estimate
        return y_pred, y_prob_ts

    return predict_fn


# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------

PIPELINE_REGISTRY: dict[str, tuple[str, callable]] = {
    "mdm": ("MDM (Riemann)", make_mdm_predict_fn),
    "ts_svm": ("TS+SVM", make_ts_svm_predict_fn),
    "ts_logreg": ("TS+LogReg", make_ts_logreg_predict_fn),
    "fbcsp_svm": ("FBCSP+SVM", make_fbcsp_svm_predict_fn),
    "csp_svm": ("CSP+SVM", make_csp_svm_predict_fn),
    "ensemble": ("Ensemble (MDM+TS+CSP)", make_ensemble_predict_fn),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_results_table(
    results: dict[str, tuple[CVResult, CVResult]],
) -> None:
    """Print a formatted comparison table of all pipelines."""
    print("\n" + "=" * 80)
    print("  IMPROVED BASELINES: COMPARATIVE RESULTS")
    print("=" * 80)
    print(
        f"\n{'Pipeline':<28} {'Within-Subj':>11} {'±':>2} {'Std':>6} "
        f"{'LOSO':>11} {'±':>2} {'Std':>6} {'Kappa':>7}"
    )
    print("-" * 80)

    for pipeline_key, (within_result, loso_result) in results.items():
        name = PIPELINE_REGISTRY[pipeline_key][0]
        print(
            f"{name:<28} "
            f"{within_result.mean_accuracy:>10.2f}% "
            f"{'':>2} "
            f"{within_result.std_accuracy:>5.2f}% "
            f"{loso_result.mean_accuracy:>10.2f}% "
            f"{'':>2} "
            f"{loso_result.std_accuracy:>5.2f}% "
            f"{within_result.mean_kappa:>7.4f}"
        )
    print("=" * 80)

    # Print LOSO per-subject breakdown for the best pipeline
    if results:
        best_key = max(results, key=lambda k: results[k][1].mean_accuracy)
        best_name = PIPELINE_REGISTRY[best_key][0]
        best_loso = results[best_key][1]
        if best_loso.per_subject_accuracy:
            print(f"\nBest LOSO pipeline: {best_name}")
            print("LOSO per-subject accuracy:")
            for subj, acc in sorted(best_loso.per_subject_accuracy.items()):
                bar = "#" * int(acc / 5)
                print(f"  S{subj:02d}: {acc:6.2f}%  {bar}")
    print()


def save_results(
    results: dict[str, tuple[CVResult, CVResult]],
    output_path: Path,
) -> None:
    """Save all pipeline results to a JSON file."""
    output = {}
    for pipeline_key, (within_result, loso_result) in results.items():
        name = PIPELINE_REGISTRY[pipeline_key][0]
        output[pipeline_key] = {
            "model": name,
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
        json.dump(output, f, indent=2)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved baselines comparison")
    parser.add_argument(
        "--data",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    parser.add_argument("--data-dir", type=str, default="~/mne_data", help="MNE data dir")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--n-subjects", type=int, default=9, help="Subjects (synthetic only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(PIPELINE_REGISTRY.keys()),
        choices=list(PIPELINE_REGISTRY.keys()),
        help="Which pipelines to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_improved.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=128.0,
        help="Sampling frequency for FBCSP filter bank (default: 128.0)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info("=" * 60)
    logger.info("Improved Baselines Comparison")
    logger.info("  Data:       %s", args.data)
    logger.info("  Pipelines:  %s", ", ".join(args.pipelines))
    logger.info("  CV folds:   %d", args.n_folds)
    logger.info("  Seed:       %d", args.seed)
    logger.info("=" * 60)

    # --- Load data ---
    if args.data == "synthetic":
        logger.info("Generating synthetic data (%d subjects)...", args.n_subjects)
        subject_data = make_synthetic_subject_data(n_subjects=args.n_subjects)
    else:
        subject_data = load_real_data(args.data_dir)

    if not subject_data:
        logger.error("No data loaded. Exiting.")
        return

    # --- Run each pipeline ---
    results: dict[str, tuple[CVResult, CVResult]] = {}

    for pipeline_key in args.pipelines:
        model_name, factory_fn = PIPELINE_REGISTRY[pipeline_key]
        logger.info("\n" + "-" * 50)
        logger.info("Running pipeline: %s", model_name)
        logger.info("-" * 50)

        # Build predict_fn with appropriate kwargs
        if pipeline_key == "fbcsp_svm":
            predict_fn = factory_fn(sfreq=args.sfreq)
        else:
            predict_fn = factory_fn()

        # Within-subject CV
        logger.info("  Within-subject %d-fold CV...", args.n_folds)
        t0 = time.time()
        within_result = within_subject_cv_all(
            subject_data,
            predict_fn,
            model_name=model_name,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        t_within = time.time() - t0
        logger.info(
            "  -> %.2f%% ± %.2f%% (%.1fs)",
            within_result.mean_accuracy,
            within_result.std_accuracy,
            t_within,
        )

        # LOSO CV
        logger.info("  LOSO CV...")
        t0 = time.time()
        loso_result = loso_cv(subject_data, predict_fn, model_name=model_name)
        t_loso = time.time() - t0
        logger.info(
            "  -> %.2f%% ± %.2f%% (%.1fs)",
            loso_result.mean_accuracy,
            loso_result.std_accuracy,
            t_loso,
        )

        results[pipeline_key] = (within_result, loso_result)

    # --- Print and save ---
    print_results_table(results)
    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
