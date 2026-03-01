"""Baseline B: Riemannian + LDA (within-subject CV + LOSO).

Loads BCI IV-2a epochs from the .npz cache.
Computes Riemannian tangent-space features (OAS covariance, Riemannian
metric) and feeds them into LDA (lsqr/auto shrinkage). Runs 5-fold
within-subject CV and a Leave-One-Subject-Out CV.

Prerequisite: Download must have been run first.

Output: <run-dir>/results/real_baseline_b_riemannian.json

Usage::

    uv run python scripts/pipeline/stage_03_baseline_b.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_03_baseline_b.py \\
        --run-dir runs/my_run --processed-dir data/processed --n-folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bci.utils.logging import setup_stage_logging


def make_predict_fn(estimator: str = "oas", metric: str = "riemann"):
    """Return a predict_fn closure for Riemannian + LDA.

    The returned function has signature::

        predict_fn(X_train, y_train, X_test) -> (y_pred, y_prob)

    Parameters
    ----------
    estimator:
        Covariance estimator passed to
        :class:`bci.features.riemannian.RiemannianFeatureExtractor`.
    metric:
        Riemannian metric (e.g. ``"riemann"``, ``"logeuclid"``).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from bci.features.riemannian import RiemannianFeatureExtractor

    def predict_fn(X_train, y_train, X_test):
        riemann = RiemannianFeatureExtractor(estimator=estimator, metric=metric)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        feats_train = riemann.fit_transform(X_train, y_train)
        feats_test = riemann.transform(X_test)
        lda.fit(feats_train, y_train)
        return lda.predict(feats_test), lda.predict_proba(feats_test)

    return predict_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline B – Riemannian + LDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Run directory (results saved here)")
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Root of processed .npz cache (default: data/processed/)",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_stage_logging(run_dir, "baseline_riemann", "baseline_riemann.log")

    out_path = run_dir / "results" / "real_baseline_b_riemannian.json"
    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from bci.data.download import load_all_subjects
    from bci.features.riemannian import RiemannianFeatureExtractor
    import numpy as np

    from bci.training.cross_validation import CVResult, FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.splits import get_or_create_splits
    from bci.utils.seed import set_seed

    MODEL_NAME = "Riemannian+LDA"
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    log.info("Loading BCI IV-2a from .npz cache...")
    try:
        subject_data, channel_names, sfreq = load_all_subjects("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run the download step first.", e)
        sys.exit(1)
    log.info(
        "Loaded %d subjects  (sfreq=%.0f Hz, %d channels)",
        len(subject_data),
        sfreq,
        len(channel_names),
    )

    def predict_fn(X_train, y_train, X_test):
        riemann = RiemannianFeatureExtractor(estimator="oas", metric="riemann")
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        feats_train = riemann.fit_transform(X_train, y_train)
        feats_test = riemann.transform(X_test)
        lda.fit(feats_train, y_train)
        return lda.predict(feats_test), lda.predict_proba(feats_test)

    set_seed(args.seed)

    split_spec = get_or_create_splits(
        run_dir=run_dir,
        dataset="bci_iv2a",
        subject_data=subject_data,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    log.info("Within-subject %d-fold CV...", split_spec.n_folds)
    t0 = time.time()
    within_folds: list[FoldResult] = []
    for sid, (X, y) in sorted(subject_data.items()):
        folds = split_spec.within_subject.get(sid, [])
        for fold_idx, fold in enumerate(folds):
            train_idx = np.array(fold["train_idx"], dtype=int)
            test_idx = np.array(fold["test_idx"], dtype=int)
            y_pred, y_prob = predict_fn(X[train_idx], y[train_idx], X[test_idx])
            m = compute_metrics(y[test_idx], y_pred, y_prob)
            within_folds.append(
                FoldResult(
                    fold=fold_idx,
                    subject=sid,
                    accuracy=m["accuracy"],
                    kappa=m["kappa"],
                    f1_macro=m["f1_macro"],
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    y_true=y[test_idx],
                    y_pred=y_pred,
                    y_prob=y_prob,
                )
            )
    within = CVResult(strategy="within_subject", model_name=MODEL_NAME, folds=within_folds)
    log.info(
        "Within-subject done in %.1fs: %.2f%% ± %.2f%%",
        time.time() - t0,
        within.mean_accuracy,
        within.std_accuracy,
    )

    log.info("LOSO CV...")
    t0 = time.time()
    loso_folds: list[FoldResult] = []
    for fold_idx, test_sid in enumerate(split_spec.loso_subjects):
        train_sids = [s for s in split_spec.loso_subjects if s != test_sid]
        X_train = np.concatenate([subject_data[s][0] for s in train_sids], axis=0)
        y_train = np.concatenate([subject_data[s][1] for s in train_sids], axis=0)
        X_test, y_test = subject_data[test_sid]
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        m = compute_metrics(y_test, y_pred, y_prob)
        loso_folds.append(
            FoldResult(
                fold=fold_idx,
                subject=test_sid,
                accuracy=m["accuracy"],
                kappa=m["kappa"],
                f1_macro=m["f1_macro"],
                n_train=len(y_train),
                n_test=len(y_test),
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
            )
        )
    loso = CVResult(strategy="loso", model_name=MODEL_NAME, folds=loso_folds)
    log.info(
        "LOSO done in %.1fs: %.2f%% ± %.2f%%",
        time.time() - t0,
        loso.mean_accuracy,
        loso.std_accuracy,
    )

    results = {
        "model": MODEL_NAME,
        "within_subject": {
            "mean_accuracy": within.mean_accuracy,
            "std_accuracy": within.std_accuracy,
            "mean_kappa": within.mean_kappa,
            "mean_f1": within.mean_f1,
            "n_folds": len(within.folds),
        },
        "loso": {
            "mean_accuracy": loso.mean_accuracy,
            "std_accuracy": loso.std_accuracy,
            "mean_kappa": loso.mean_kappa,
            "mean_f1": loso.mean_f1,
            "n_folds": len(loso.folds),
            "per_subject": loso.per_subject_accuracy,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    try:
        from bci.utils.results_index import update_results_index, write_manifest

        outputs = {"within_loso": str(out_path)}
        update_results_index(run_dir, "baseline_riemann", outputs)
        write_manifest(
            run_dir,
            "baseline_riemann",
            outputs,
            meta={"n_folds": args.n_folds, "seed": args.seed},
        )
    except Exception as e:
        log.warning("Failed to update results index: %s", e)
    log.info("Saved: %s", out_path)
    log.info("Baseline B complete.")


if __name__ == "__main__":
    main()
