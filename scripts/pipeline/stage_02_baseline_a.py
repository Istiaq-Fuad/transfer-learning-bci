"""Stage 2 – Baseline A: CSP + LDA (within-subject + LOSO).

Trains a CSP spatial filter followed by a Linear Discriminant Analysis
classifier. Runs 5-fold within-subject CV across all 9 BCI IV-2a subjects
and a Leave-One-Subject-Out CV.

Output: <run-dir>/results/real_baseline_a_csp_lda.json

Usage::

    uv run python scripts/pipeline/stage_02_baseline_a.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_02_baseline_a.py --run-dir runs/my_run --n-folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: Baseline A – CSP + LDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",  required=True, help="Run directory (results saved here)")
    p.add_argument("--data-dir", default="~/mne_data", help="MNE data directory")
    p.add_argument("--n-folds",  type=int, default=5)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout)
    for lib in ("mne", "pyriemann", "timm", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_02")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "stage_02_baseline_a.log")
    fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    log.addHandler(fh)
    return log


def load_bci_iv2a(data_dir: str, log) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    log.info("Loading BCI IV-2a (sfreq=128 Hz)...")
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=128.0)
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in dataset.subject_list:
        try:
            X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[sid])
            classes = sorted(np.unique(y_labels))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[lb] for lb in y_labels], dtype=np.int64)
            subject_data[sid] = (X.astype(np.float32), y)
            log.info("  Subject %d: X=%s", sid, X.shape)
        except Exception as e:
            log.warning("  Subject %d skipped: %s", sid, e)
    log.info("Loaded %d subjects.", len(subject_data))
    return subject_data


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_logging(run_dir)

    out_path = run_dir / "results" / "real_baseline_a_csp_lda.json"
    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from bci.features.csp import CSPFeatureExtractor
    from bci.training.cross_validation import loso_cv, within_subject_cv_all
    from bci.utils.seed import set_seed

    MODEL_NAME = "CSP+LDA"

    subject_data = load_bci_iv2a(args.data_dir, log)
    if not subject_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    def predict_fn(X_train, y_train, X_test):
        csp = CSPFeatureExtractor(n_components=6, reg="ledoit_wolf")
        lda = LinearDiscriminantAnalysis()
        feats_train = csp.fit_transform(X_train, y_train)
        feats_test  = csp.transform(X_test)
        lda.fit(feats_train, y_train)
        return lda.predict(feats_test), lda.predict_proba(feats_test)

    set_seed(args.seed)

    log.info("Within-subject %d-fold CV...", args.n_folds)
    t0 = time.time()
    within = within_subject_cv_all(
        subject_data, predict_fn, model_name=MODEL_NAME,
        n_folds=args.n_folds, seed=args.seed,
    )
    log.info("Within-subject done in %.1fs: %.2f%% ± %.2f%%",
             time.time() - t0, within.mean_accuracy, within.std_accuracy)

    log.info("LOSO CV...")
    t0 = time.time()
    loso = loso_cv(subject_data, predict_fn, model_name=MODEL_NAME)
    log.info("LOSO done in %.1fs: %.2f%% ± %.2f%%",
             time.time() - t0, loso.mean_accuracy, loso.std_accuracy)

    results = {
        "model": MODEL_NAME,
        "within_subject": {
            "mean_accuracy": within.mean_accuracy,
            "std_accuracy":  within.std_accuracy,
            "mean_kappa":    within.mean_kappa,
            "mean_f1":       within.mean_f1,
            "n_folds":       len(within.folds),
        },
        "loso": {
            "mean_accuracy": loso.mean_accuracy,
            "std_accuracy":  loso.std_accuracy,
            "mean_kappa":    loso.mean_kappa,
            "mean_f1":       loso.mean_f1,
            "n_folds":       len(loso.folds),
            "per_subject":   loso.per_subject_accuracy,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_path)
    log.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
