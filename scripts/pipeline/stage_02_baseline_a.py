"""Stage 02 – Baseline A: CSP + LDA (within-subject CV + LOSO).

Loads BCI IV-2a epochs from the .npz cache written by Stage 01.
Trains a CSP spatial filter (Ledoit-Wolf, 6 components) followed by LDA
(lsqr/auto shrinkage). Runs 5-fold within-subject CV across all 9 subjects
and a Leave-One-Subject-Out CV.

Prerequisite: Stage 01 must have been run first.

Output: <run-dir>/results/real_baseline_a_csp_lda.json

Usage::

    uv run python scripts/pipeline/stage_02_baseline_a.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_02_baseline_a.py \\
        --run-dir runs/my_run --processed-dir data/processed --n-folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bci.utils.logging import setup_stage_logging


def make_predict_fn(n_components: int = 6, reg: str = "ledoit_wolf"):
    """Return a predict_fn closure for CSP + LDA.

    The returned function has signature::

        predict_fn(X_train, y_train, X_test) -> (y_pred, y_prob)

    Parameters
    ----------
    n_components:
        Number of CSP spatial filters.
    reg:
        Regularisation method passed to :class:`bci.features.csp.CSPFeatureExtractor`.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from bci.features.csp import CSPFeatureExtractor

    def predict_fn(X_train, y_train, X_test):
        csp = CSPFeatureExtractor(n_components=n_components, reg=reg)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        feats_train = csp.fit_transform(X_train, y_train)
        feats_test = csp.transform(X_test)
        lda.fit(feats_train, y_train)
        return lda.predict(feats_test), lda.predict_proba(feats_test)

    return predict_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 02: Baseline A – CSP + LDA.",
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
    log = setup_stage_logging(run_dir, "stage_02", "stage_02_baseline_a.log")

    out_path = run_dir / "results" / "real_baseline_a_csp_lda.json"
    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from bci.data.download import load_all_subjects
    from bci.features.csp import CSPFeatureExtractor
    from bci.training.cross_validation import loso_cv, within_subject_cv_all
    from bci.utils.seed import set_seed

    MODEL_NAME = "CSP+LDA"
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    log.info("Loading BCI IV-2a from .npz cache...")
    try:
        subject_data, channel_names, sfreq = load_all_subjects("bci_iv2a", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run Stage 01 first.", e)
        sys.exit(1)
    log.info(
        "Loaded %d subjects  (sfreq=%.0f Hz, %d channels)",
        len(subject_data),
        sfreq,
        len(channel_names),
    )

    def predict_fn(X_train, y_train, X_test):
        csp = CSPFeatureExtractor(n_components=6, reg="ledoit_wolf")
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        feats_train = csp.fit_transform(X_train, y_train)
        feats_test = csp.transform(X_test)
        lda.fit(feats_train, y_train)
        return lda.predict(feats_test), lda.predict_proba(feats_test)

    set_seed(args.seed)

    log.info("Within-subject %d-fold CV...", args.n_folds)
    t0 = time.time()
    within = within_subject_cv_all(
        subject_data,
        predict_fn,
        model_name=MODEL_NAME,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    log.info(
        "Within-subject done in %.1fs: %.2f%% ± %.2f%%",
        time.time() - t0,
        within.mean_accuracy,
        within.std_accuracy,
    )

    log.info("LOSO CV...")
    t0 = time.time()
    loso = loso_cv(subject_data, predict_fn, model_name=MODEL_NAME)
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
    log.info("Saved: %s", out_path)
    log.info("Stage 02 complete.")


if __name__ == "__main__":
    main()
