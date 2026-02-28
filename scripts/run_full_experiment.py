"""Overnight Full Experiment Runner.

Runs the entire thesis pipeline from start to finish on a single GPU machine.
Stages are skipped automatically if their result file already exists, making
the script resumable after crashes.

Stages:
    1.  Download / verify BCI IV-2a and PhysioNet datasets
    2.  Baseline A – CSP + LDA (within-subject + LOSO)
    3.  Baseline B – Riemannian + LDA (within-subject + LOSO)
    4.  Baseline C – CWT + ViT-Tiny (within-subject only)
    5.  Phase 2 – Dual-branch, attention fusion (within-subject + LOSO)
    6.  Phase 2 – Dual-branch, gated fusion (within-subject)
    7.  Phase 3a – Pretrain ViT on PhysioNet (real data)
    8.  Phase 3b – Finetune comparison (scratch / imagenet / transfer)
    9.  Phase 3c – Reduced-data experiment
    10. Phase 4  – Compile, visualize, statistical tests

All outputs go to a timestamped directory, e.g.::
    runs/2024-01-15_143022/
        results/
            real_baseline_a_csp_lda.json
            real_baseline_b_riemannian.json
            ...
            phase4_summary.json
            phase4_stats.json
        checkpoints/
            vit_pretrained_physionet.pt
        figures/
            fig1_spectrogram_examples.png
            ...
        experiment.log

Usage (GPU machine, overnight)::
    uv run python scripts/run_full_experiment.py

    # Skip already-done stages (safe to re-run after crash)
    uv run python scripts/run_full_experiment.py --run-dir runs/2024-01-15_143022

    # Dry run: just download data and print plan
    uv run python scripts/run_full_experiment.py --dry-run

    # Override some hyperparameters
    uv run python scripts/run_full_experiment.py --epochs 30 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging setup: tee to file + console
# ---------------------------------------------------------------------------


class _TeeHandler(logging.Handler):
    """Write log records to a file handle."""

    def __init__(self, fh):
        super().__init__()
        self._fh = fh

    def emit(self, record):
        try:
            self._fh.write(self.format(record) + "\n")
            self._fh.flush()
        except Exception:
            pass


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    # File
    fh_raw = open(log_path, "a", encoding="utf-8")
    fh = _TeeHandler(fh_raw)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    # Silence noisy libraries
    for lib in ("mne", "pyriemann", "timm", "matplotlib"):
        logging.getLogger(lib).setLevel(logging.ERROR)

    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------


def banner(msg: str, char: str = "=", width: int = 70) -> None:
    line = char * width
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}\n", flush=True)


def stage_banner(n: int, total: int, name: str) -> None:
    banner(f"STAGE {n}/{total}: {name}")


def skip_banner(name: str, path: Path) -> None:
    banner(f"SKIP {name}  (result already exists: {path.name})", char="-")


# ---------------------------------------------------------------------------
# Shared data loader (BCI IV-2a)
# ---------------------------------------------------------------------------


def load_bci_iv2a(
    data_dir: str,
    sfreq: float = 128.0,
    logger: logging.Logger | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load BCI Competition IV-2a via MOABB.  Returns {subject_id: (X, y)}."""
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    log = logger or logging.getLogger(__name__)
    log.info("Loading BCI IV-2a from %s (sfreq=%.0f Hz)...", data_dir, sfreq)

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=sfreq)

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in dataset.subject_list:
        try:
            X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[sid])
            classes = sorted(np.unique(y_labels))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[lb] for lb in y_labels], dtype=np.int64)
            subject_data[sid] = (X.astype(np.float32), y)
            log.info("  Subject %d: X=%s  classes=%s", sid, X.shape, classes)
        except Exception as e:
            log.warning("  Skipping subject %d: %s", sid, e)

    log.info("Loaded %d subjects from BCI IV-2a.", len(subject_data))
    return subject_data


def load_physionet(
    n_subjects: int | None,
    sfreq: float = 128.0,
    logger: logging.Logger | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str], float]:
    """Load PhysioNet MMIDB via MOABB.  Returns (subject_data, channel_names, sfreq)."""
    import mne
    from moabb.datasets import PhysionetMI
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    log = logger or logging.getLogger(__name__)
    log.info("Loading PhysioNet MMIDB (n_subjects=%s)...", n_subjects)

    dataset = PhysionetMI()
    paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=sfreq)

    subjects = dataset.subject_list
    if n_subjects is not None:
        subjects = subjects[:n_subjects]

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in subjects:
        try:
            X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[sid])
            classes = sorted(np.unique(y_labels))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[lb] for lb in y_labels], dtype=np.int64)
            subject_data[sid] = (X.astype(np.float32), y)
            log.info("  Subject %d: X=%s", sid, X.shape)
        except Exception as e:
            log.warning("  Skipping subject %d: %s", sid, e)

    log.info("Loaded %d subjects from PhysioNet.", len(subject_data))
    channel_names = ["C3", "Cz", "C4"]
    return subject_data, channel_names, sfreq


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

# ── Stage 1: Download / verify datasets ────────────────────────────────────


def stage_download(args, run_dir: Path, log) -> None:
    """Trigger dataset download by loading 1 subject from each dataset."""
    banner("Downloading / verifying datasets")

    log.info("Checking BCI IV-2a...")
    try:
        data = load_bci_iv2a(args.data_dir, logger=log)
        log.info("BCI IV-2a OK: %d subjects available.", len(data))
    except Exception as e:
        log.error("BCI IV-2a download/load failed: %s", e)
        raise

    log.info("Checking PhysioNet MMIDB...")
    try:
        pdata, _, _ = load_physionet(n_subjects=2, logger=log)
        log.info("PhysioNet OK: %d subjects verified.", len(pdata))
    except Exception as e:
        log.error("PhysioNet download/load failed: %s", e)
        raise


# ── Stage 2: Baseline A – CSP + LDA ────────────────────────────────────────


def stage_baseline_a(
    subject_data: dict,
    run_dir: Path,
    n_folds: int,
    seed: int,
    log,
) -> Path:
    out_path = run_dir / "results" / "real_baseline_a_csp_lda.json"
    if out_path.exists():
        skip_banner("Baseline A: CSP+LDA", out_path)
        return out_path

    import time
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from bci.features.csp import CSPFeatureExtractor
    from bci.training.cross_validation import CVResult, loso_cv, within_subject_cv_all
    from bci.utils.seed import set_seed

    MODEL_NAME = "CSP+LDA"

    def predict_fn(X_train, y_train, X_test):
        csp = CSPFeatureExtractor(n_components=6, reg="ledoit_wolf")
        lda = LinearDiscriminantAnalysis()
        feats_train = csp.fit_transform(X_train, y_train)
        feats_test = csp.transform(X_test)
        lda.fit(feats_train, y_train)
        return lda.predict(feats_test), lda.predict_proba(feats_test)

    set_seed(seed)

    log.info("Within-subject %d-fold CV...", n_folds)
    t0 = time.time()
    within = within_subject_cv_all(
        subject_data,
        predict_fn,
        model_name=MODEL_NAME,
        n_folds=n_folds,
        seed=seed,
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
    return out_path


# ── Stage 3: Baseline B – Riemannian + LDA ─────────────────────────────────


def stage_baseline_b(
    subject_data: dict,
    run_dir: Path,
    n_folds: int,
    seed: int,
    log,
) -> Path:
    out_path = run_dir / "results" / "real_baseline_b_riemannian.json"
    if out_path.exists():
        skip_banner("Baseline B: Riemannian+LDA", out_path)
        return out_path

    import time
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from bci.features.riemannian import RiemannianFeatureExtractor
    from bci.training.cross_validation import CVResult, loso_cv, within_subject_cv_all
    from bci.utils.seed import set_seed

    MODEL_NAME = "Riemannian+LDA"

    def predict_fn(X_train, y_train, X_test):
        riemann = RiemannianFeatureExtractor(estimator="lwf", metric="riemann")
        lda = LinearDiscriminantAnalysis()
        feats_train = riemann.fit_transform(X_train, y_train)
        feats_test = riemann.transform(X_test)
        lda.fit(feats_train, y_train)
        return lda.predict(feats_test), lda.predict_proba(feats_test)

    set_seed(seed)

    log.info("Within-subject %d-fold CV...", n_folds)
    t0 = time.time()
    within = within_subject_cv_all(
        subject_data,
        predict_fn,
        model_name=MODEL_NAME,
        n_folds=n_folds,
        seed=seed,
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
    return out_path


# ── Stage 4: Baseline C – CWT + ViT-Tiny ───────────────────────────────────


def stage_baseline_c(
    subject_data: dict,
    run_dir: Path,
    n_folds: int,
    epochs: int,
    batch_size: int,
    device,
    seed: int,
    log,
) -> Path:
    out_path = run_dir / "results" / "real_baseline_c_vit.json"
    if out_path.exists():
        skip_banner("Baseline C: CWT+ViT-Tiny", out_path)
        return out_path

    import time
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    from bci.data.transforms import CWTSpectrogramTransform
    from bci.models.vit_branch import ViTBranch
    from bci.training.cross_validation import within_subject_cv_all
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig, SpectrogramConfig
    from bci.utils.seed import set_seed

    MODEL_NAME = "CWT+ViT-Tiny"
    CHANNEL_NAMES = ["C3", "Cz", "C4"]
    SFREQ = 128.0

    spec_config = SpectrogramConfig(
        wavelet="morl",
        freq_min=4.0,
        freq_max=40.0,
        n_freqs=64,
        image_size=(224, 224),
        channel_mode="rgb_c3_cz_c4",
    )
    transform = CWTSpectrogramTransform(spec_config)

    def epochs_to_imgs(X):
        hwc = transform.transform_epochs(X, CHANNEL_NAMES, SFREQ)
        return hwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    _device = torch.device(device)

    def predict_fn(X_train, y_train, X_test):
        set_seed(seed)
        imgs_train = epochs_to_imgs(X_train)
        imgs_test = epochs_to_imgs(X_test)
        train_ds = TensorDataset(torch.tensor(imgs_train), torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(
            torch.tensor(imgs_test),
            torch.tensor(np.zeros(len(X_test), dtype=np.int64), dtype=torch.long),
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

        model_config = ModelConfig(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=True,
            vit_drop_rate=0.1,
            n_classes=2,
        )
        model = ViTBranch(config=model_config, as_feature_extractor=False)
        model.freeze_backbone(unfreeze_last_n_blocks=2)

        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=1e-4,
            weight_decay=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            warmup_epochs=5,
            patience=10,
            label_smoothing=0.1,
            val_fraction=0.2,
            seed=seed,
            num_workers=0,
        )
        trainer.fit(train_ds, model_tag="baseline_c_fold")
        return trainer.predict(test_loader)

    set_seed(seed)
    log.info("Within-subject %d-fold CV (ViT)...", n_folds)
    t0 = time.time()
    within = within_subject_cv_all(
        subject_data,
        predict_fn,
        model_name=MODEL_NAME,
        n_folds=n_folds,
        seed=seed,
    )
    log.info(
        "Baseline C done in %.1fs: %.2f%% ± %.2f%%",
        time.time() - t0,
        within.mean_accuracy,
        within.std_accuracy,
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
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_path)
    return out_path


# ── Stage 5/6/7: Dual-branch (one fusion method) ───────────────────────────


def stage_dual_branch(
    fusion: str,
    strategy: str,  # "within_subject" or "loso"
    subject_data: dict,
    run_dir: Path,
    n_folds: int,
    epochs: int,
    batch_size: int,
    device,
    seed: int,
    log,
) -> Path:
    tag = f"real_dual_branch_{fusion}"
    if strategy == "loso":
        tag += "_loso"
    out_path = run_dir / "results" / f"{tag}.json"
    if out_path.exists():
        skip_banner(f"Dual-branch [{fusion}] {strategy}", out_path)
        return out_path

    import time
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader

    from bci.data.dual_branch_builder import DualBranchFoldBuilder
    from bci.models.dual_branch import DualBranchModel
    from bci.training.cross_validation import CVResult, FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import set_seed

    MODEL_NAME = "DualBranch-ViT+CSP+Riemann"

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        csp_reg="ledoit_wolf",
        riemann_estimator="lwf",
        riemann_metric="riemann",
        sfreq=128.0,
        channel_names=["C3", "Cz", "C4"],
    )

    _device = torch.device(device)

    def train_and_eval_fold(fold_idx, subject_id, X_train, y_train, X_test, y_test):
        set_seed(seed + fold_idx)
        train_ds, test_ds, math_input_dim = builder.build_fold(X_train, y_train, X_test, y_test)
        model_config = ModelConfig(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=True,
            vit_drop_rate=0.1,
            csp_n_components=6,
            math_hidden_dims=[256, 128],
            math_drop_rate=0.3,
            fusion_method=fusion,
            fused_dim=128,
            classifier_hidden_dim=64,
            n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=model_config)
        model.freeze_vit_backbone(unfreeze_last_n_blocks=2)

        def dual_fwd(batch):
            imgs, feats, labels = batch
            return model(imgs.to(_device), feats.to(_device)), labels.to(_device)

        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=1e-4,
            weight_decay=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            warmup_epochs=5,
            patience=10,
            label_smoothing=0.1,
            val_fraction=0.2,
            seed=seed,
            num_workers=0,
        )
        trainer.fit(train_ds, forward_fn=dual_fwd, model_tag=f"dual_{fusion}_f{fold_idx}")

        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
        y_pred, y_prob = trainer.predict(test_loader, forward_fn=dual_fwd)
        m = compute_metrics(y_test, y_pred, y_prob)
        fr = FoldResult(
            fold=fold_idx,
            subject=subject_id,
            accuracy=m["accuracy"],
            kappa=m["kappa"],
            f1_macro=m["f1_macro"],
            n_train=len(y_train),
            n_test=len(y_test),
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        log.info(
            "  Fold %d [S%s]: acc=%.2f%%  kappa=%.3f",
            fold_idx,
            f"{subject_id:02d}" if subject_id is not None else "?",
            fr.accuracy,
            fr.kappa,
        )
        return fr

    t0 = time.time()
    all_folds: list[FoldResult] = []

    if strategy == "within_subject":
        fold_counter = 0
        for sid, (X, y) in sorted(subject_data.items()):
            log.info("Subject %d (%d trials)...", sid, len(y))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                fr = train_and_eval_fold(
                    fold_counter,
                    sid,
                    X[train_idx],
                    y[train_idx],
                    X[test_idx],
                    y[test_idx],
                )
                all_folds.append(fr)
                fold_counter += 1
    else:  # loso
        subjects = sorted(subject_data.keys())
        for fold_idx, test_sid in enumerate(subjects):
            train_Xs = [subject_data[s][0] for s in subjects if s != test_sid]
            train_ys = [subject_data[s][1] for s in subjects if s != test_sid]
            X_train = np.concatenate(train_Xs, axis=0)
            y_train = np.concatenate(train_ys, axis=0)
            X_test, y_test = subject_data[test_sid]
            log.info("LOSO fold %d/%d: test=S%02d", fold_idx + 1, len(subjects), test_sid)
            fr = train_and_eval_fold(
                fold_idx,
                test_sid,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            all_folds.append(fr)

    elapsed = time.time() - t0
    result = CVResult(strategy=strategy, model_name=MODEL_NAME, folds=all_folds)
    log.info(
        "Dual-branch [%s] %s done in %.1fs: %.2f%% ± %.2f%%",
        fusion,
        strategy,
        elapsed,
        result.mean_accuracy,
        result.std_accuracy,
    )

    data = {
        "model": MODEL_NAME,
        "fusion": fusion,
        "strategy": strategy,
        "mean_accuracy": result.mean_accuracy,
        "std_accuracy": result.std_accuracy,
        "mean_kappa": result.mean_kappa,
        "mean_f1": result.mean_f1,
        "n_folds": len(all_folds),
        "per_subject": result.per_subject_accuracy,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved: %s", out_path)
    return out_path


# ── Stage 8: Pretrain ViT on PhysioNet ─────────────────────────────────────


def stage_pretrain(
    run_dir: Path,
    n_subjects: int,
    epochs: int,
    batch_size: int,
    device,
    seed: int,
    log,
) -> Path:
    checkpoint_path = run_dir / "checkpoints" / "vit_pretrained_physionet.pt"
    out_path = run_dir / "results" / "real_pretrain_physionet.json"
    if checkpoint_path.exists() and out_path.exists():
        skip_banner("Phase 3a: Pretrain ViT on PhysioNet", checkpoint_path)
        return checkpoint_path

    import time
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from bci.data.transforms import CWTSpectrogramTransform
    from bci.models.vit_branch import ViTBranch
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig, SpectrogramConfig
    from bci.utils.seed import set_seed

    set_seed(seed)

    source_data, channel_names, sfreq = load_physionet(n_subjects=n_subjects, logger=log)
    if not source_data:
        raise RuntimeError("PhysioNet data load failed — cannot pretrain.")

    spec_config = SpectrogramConfig(
        wavelet="morl",
        freq_min=4.0,
        freq_max=40.0,
        n_freqs=64,
        image_size=(224, 224),
        channel_mode="rgb_c3_cz_c4",
    )
    transform = CWTSpectrogramTransform(spec_config)

    def to_imgs(X):
        hwc = transform.transform_epochs(X, channel_names, sfreq)
        return hwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    # Pool all subjects
    all_X = np.concatenate([X for X, _ in source_data.values()], axis=0)
    all_y = np.concatenate([y for _, y in source_data.values()], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_y))
    all_X, all_y = all_X[idx], all_y[idx]
    log.info("Pooled source: %d trials", len(all_y))

    n_val = max(1, int(len(all_y) * 0.15))
    n_train = len(all_y) - n_val
    log.info("Generating CWT spectrograms for %d train trials...", n_train)
    imgs_train = to_imgs(all_X[:n_train])
    imgs_val = to_imgs(all_X[n_train:])
    train_ds = TensorDataset(
        torch.tensor(imgs_train), torch.tensor(all_y[:n_train], dtype=torch.long)
    )
    val_ds = TensorDataset(torch.tensor(imgs_val), torch.tensor(all_y[n_train:], dtype=torch.long))

    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=True,
        vit_drop_rate=0.1,
        n_classes=2,
    )
    model = ViTBranch(config=model_config, as_feature_extractor=False)
    _device = torch.device(device)

    def fwd(batch):
        imgs, labels = batch
        return model(imgs.to(_device)), labels.to(_device)

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=batch_size,
        warmup_epochs=5,
        patience=10,
        label_smoothing=0.1,
        val_fraction=0.2,
        seed=seed,
        num_workers=0,
    )
    t0 = time.time()
    trainer.fit(train_ds, forward_fn=fwd, model_tag="vit_pretrain")
    elapsed = time.time() - t0
    log.info("Pretraining done in %.1fs", elapsed)

    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(val_loader, forward_fn=fwd)
    metrics = compute_metrics(all_y[n_train:], y_pred, y_prob)
    log.info("Pretrain val: %.2f%%  kappa=%.3f", metrics["accuracy"], metrics["kappa"])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    log.info("Checkpoint saved: %s", checkpoint_path)

    pretrain_results = {
        "phase": "pretrain",
        "source": "real_physionet",
        "n_subjects": len(source_data),
        "val_accuracy": metrics["accuracy"],
        "val_kappa": metrics["kappa"],
        "val_f1": metrics["f1_macro"],
        "n_source_trials": int(len(all_y)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(checkpoint_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pretrain_results, f, indent=2)
    log.info("Saved: %s", out_path)
    return checkpoint_path


# ── Stage 9: Finetune comparison (scratch / imagenet / transfer) ────────────


def stage_finetune(
    subject_data: dict,
    run_dir: Path,
    checkpoint_path: Path,
    conditions: list[str],
    n_folds: int,
    epochs: int,
    batch_size: int,
    device,
    seed: int,
    log,
) -> dict[str, Path]:
    """Run all three finetune conditions.  Returns {condition: out_path}."""
    import time
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader

    from bci.data.dual_branch_builder import DualBranchFoldBuilder
    from bci.models.dual_branch import DualBranchModel
    from bci.training.cross_validation import CVResult, FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import set_seed

    MODEL_NAME = "DualBranch-Transfer"
    FUSION = "attention"

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        csp_reg="ledoit_wolf",
        riemann_estimator="lwf",
        riemann_metric="riemann",
        sfreq=128.0,
        channel_names=["C3", "Cz", "C4"],
    )
    _device = torch.device(device)

    def build_model(condition, math_input_dim):
        use_imagenet = condition in ("imagenet", "transfer")
        model_config = ModelConfig(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=use_imagenet,
            vit_drop_rate=0.1,
            csp_n_components=6,
            math_hidden_dims=[256, 128],
            math_drop_rate=0.3,
            fusion_method=FUSION,
            fused_dim=128,
            classifier_hidden_dim=64,
            n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=model_config)
        if condition == "transfer":
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            backbone_state = {k: v for k, v in ckpt.items() if not k.startswith("backbone.head")}
            model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
            model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        elif condition == "imagenet":
            model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        return model

    out_paths: dict[str, Path] = {}

    for condition in conditions:
        out_path = run_dir / "results" / f"real_finetune_{condition}.json"
        out_paths[condition] = out_path
        if out_path.exists():
            skip_banner(f"Phase 3b: Finetune [{condition}]", out_path)
            continue

        log.info("--- Finetune condition: %s ---", condition.upper())
        all_folds: list[FoldResult] = []
        fold_counter = 0
        t0 = time.time()

        for sid, (X, y) in sorted(subject_data.items()):
            log.info("  Subject %d (%d trials)...", sid, len(y))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                set_seed(seed + fold_counter)
                train_ds, test_ds, math_input_dim = builder.build_fold(
                    X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                )
                model = build_model(condition, math_input_dim)

                def dual_fwd(batch, _m=model):
                    imgs, feats, labels = batch
                    return _m(imgs.to(_device), feats.to(_device)), labels.to(_device)

                trainer = Trainer(
                    model=model,
                    device=device,
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    epochs=epochs,
                    batch_size=batch_size,
                    warmup_epochs=5,
                    patience=10,
                    label_smoothing=0.1,
                    val_fraction=0.2,
                    seed=seed,
                    num_workers=0,
                )
                trainer.fit(
                    train_ds,
                    forward_fn=dual_fwd,
                    model_tag=f"{condition}_f{fold_counter}",
                )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
                )
                y_pred, y_prob = trainer.predict(test_loader, forward_fn=dual_fwd)
                m = compute_metrics(y[test_idx], y_pred, y_prob)
                fr = FoldResult(
                    fold=fold_counter,
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
                log.info(
                    "  [%s] Fold %d S%02d: acc=%.2f%%  kappa=%.3f",
                    condition.upper(),
                    fold_counter,
                    sid,
                    fr.accuracy,
                    fr.kappa,
                )
                all_folds.append(fr)
                fold_counter += 1

        result = CVResult(
            strategy="within_subject",
            model_name=f"{MODEL_NAME}-{condition}",
            folds=all_folds,
        )
        elapsed = time.time() - t0
        log.info(
            "[%s] Done in %.1fs: %.2f%% ± %.2f%%",
            condition.upper(),
            elapsed,
            result.mean_accuracy,
            result.std_accuracy,
        )

        data = {
            "model": f"{MODEL_NAME}-{condition}",
            "condition": condition,
            "strategy": "within_subject",
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "mean_kappa": result.mean_kappa,
            "mean_f1": result.mean_f1,
            "n_folds": len(all_folds),
            "per_subject": result.per_subject_accuracy,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Saved: %s", out_path)

    return out_paths


# ── Stage 10: Reduced-data experiment ──────────────────────────────────────


def stage_reduced_data(
    subject_data: dict,
    run_dir: Path,
    checkpoint_path: Path,
    fractions: list[float],
    n_repeats: int,
    n_folds: int,
    epochs: int,
    batch_size: int,
    device,
    seed: int,
    log,
) -> Path:
    out_path = run_dir / "results" / "real_reduced_data_results.json"
    if out_path.exists():
        skip_banner("Phase 3c: Reduced-data experiment", out_path)
        return out_path

    import time
    import torch
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from torch.utils.data import DataLoader

    from bci.data.dual_branch_builder import DualBranchFoldBuilder
    from bci.models.dual_branch import DualBranchModel
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import set_seed

    builder = DualBranchFoldBuilder(
        csp_n_components=6,
        csp_reg="ledoit_wolf",
        riemann_estimator="lwf",
        riemann_metric="riemann",
        sfreq=128.0,
        channel_names=["C3", "Cz", "C4"],
    )
    _device = torch.device(device)

    def build_model(condition, math_input_dim, ckpt_path):
        use_imagenet = condition in ("imagenet", "transfer")
        model_config = ModelConfig(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=use_imagenet,
            vit_drop_rate=0.1,
            csp_n_components=6,
            math_hidden_dims=[256, 128],
            math_drop_rate=0.3,
            fusion_method="attention",
            fused_dim=128,
            classifier_hidden_dim=64,
            n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=model_config)
        if condition == "transfer":
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            backbone_state = {k: v for k, v in ckpt.items() if not k.startswith("backbone.head")}
            model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
            model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        return model

    conditions = ["scratch", "transfer"]
    results: dict[str, dict] = {c: {} for c in conditions}

    t_total = time.time()

    for fraction in fractions:
        frac_str = f"{fraction:.2f}"
        log.info("=== Fraction %.0f%% ===", fraction * 100)
        cond_accs: dict[str, list[float]] = {c: [] for c in conditions}

        for sid, (X, y) in sorted(subject_data.items()):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train_full, X_test = X[train_idx], X[test_idx]
                y_train_full, y_test = y[train_idx], y[test_idx]

                for rep in range(n_repeats):
                    trial_seed = seed + sid * 1000 + fold_i * 100 + rep

                    for condition in conditions:
                        set_seed(trial_seed)

                        # Subsample training data
                        if fraction < 1.0:
                            n_keep = max(2, int(len(y_train_full) * fraction))
                            sss = StratifiedShuffleSplit(
                                n_splits=1, train_size=n_keep, random_state=trial_seed
                            )
                            keep_idx, _ = next(sss.split(X_train_full, y_train_full))
                            X_train = X_train_full[keep_idx]
                            y_train = y_train_full[keep_idx]
                        else:
                            X_train, y_train = X_train_full, y_train_full

                        try:
                            train_ds, test_ds, math_input_dim = builder.build_fold(
                                X_train, y_train, X_test, y_test
                            )
                            model = build_model(condition, math_input_dim, checkpoint_path)

                            def dual_fwd(batch, _m=model):
                                imgs, feats, labels = batch
                                return (
                                    _m(imgs.to(_device), feats.to(_device)),
                                    labels.to(_device),
                                )

                            trainer = Trainer(
                                model=model,
                                device=device,
                                learning_rate=1e-4,
                                weight_decay=1e-4,
                                epochs=epochs,
                                batch_size=batch_size,
                                warmup_epochs=3,
                                patience=8,
                                label_smoothing=0.1,
                                val_fraction=0.2,
                                seed=trial_seed,
                                num_workers=0,
                            )
                            trainer.fit(
                                train_ds,
                                forward_fn=dual_fwd,
                                model_tag=f"{condition}_f{fraction:.0%}",
                            )
                            test_loader = DataLoader(
                                test_ds,
                                batch_size=batch_size * 2,
                                shuffle=False,
                                num_workers=0,
                            )
                            y_pred, y_prob = trainer.predict(test_loader, forward_fn=dual_fwd)
                            m = compute_metrics(y_test, y_pred, y_prob)
                            acc = m["accuracy"]
                        except Exception as e:
                            log.warning(
                                "Trial failed (S%d fold%d rep%d %s): %s",
                                sid,
                                fold_i,
                                rep,
                                condition,
                                e,
                            )
                            acc = float("nan")

                        cond_accs[condition].append(acc)
                        log.info(
                            "  %s | S%02d fold%d rep%d | frac=%.0f%% | acc=%.2f%%",
                            condition.upper(),
                            sid,
                            fold_i,
                            rep,
                            fraction * 100,
                            acc,
                        )

        for condition in conditions:
            accs = [a for a in cond_accs[condition] if not np.isnan(a)]
            results[condition][frac_str] = {
                "fraction": fraction,
                "mean": float(np.mean(accs)) if accs else float("nan"),
                "std": float(np.std(accs)) if accs else float("nan"),
                "n_runs": len(accs),
                "runs": accs,
            }
            log.info(
                "  %s @ %.0f%%: %.2f%% ± %.2f%% (n=%d)",
                condition.upper(),
                fraction * 100,
                results[condition][frac_str]["mean"],
                results[condition][frac_str]["std"],
                len(accs),
            )

    elapsed = time.time() - t_total
    log.info("Reduced-data experiment done in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # Reformat to match compile script expectations
    summary: dict[str, dict] = {}
    for cond, frac_data in results.items():
        summary[cond] = {
            frac_str: {
                "fraction_pct": round(d["fraction"] * 100),
                "mean_accuracy": round(d["mean"], 4),
                "std_accuracy": round(d["std"], 4),
                "n_runs": d["n_runs"],
            }
            for frac_str, d in frac_data.items()
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"fractions": fractions, "results": summary}, f, indent=2)
    log.info("Saved: %s", out_path)
    return out_path


# ── Stage 11: Phase 4 – compile + visualize + stats ────────────────────────


def stage_phase4(run_dir: Path, log) -> None:
    """Run all three phase 4 analysis steps."""
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    summary_path = results_dir / "phase4_summary.json"

    # Step 4a: compile
    compile_path = Path(__file__).parent / "phase4_compile_results.py"
    import subprocess, sys

    log.info("Running phase4_compile_results.py...")
    result = subprocess.run(
        [
            sys.executable,
            str(compile_path),
            "--results-dir",
            str(results_dir),
            "--output",
            str(summary_path),
            "--prefix",
            "real_",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        log.warning("phase4_compile_results.py exited with code %d", result.returncode)

    # Step 4b: visualize
    viz_path = Path(__file__).parent / "phase4_visualize.py"
    log.info("Running phase4_visualize.py...")
    result = subprocess.run(
        [
            sys.executable,
            str(viz_path),
            "--summary",
            str(summary_path),
            "--output-dir",
            str(figures_dir),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        log.warning("phase4_visualize.py exited with code %d", result.returncode)

    # Step 4c: stats
    stats_path = Path(__file__).parent / "phase4_stats.py"
    log.info("Running phase4_stats.py...")
    result = subprocess.run(
        [
            sys.executable,
            str(stats_path),
            "--summary",
            str(summary_path),
            "--output",
            str(results_dir / "phase4_stats.json"),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        log.warning("phase4_stats.py exited with code %d", result.returncode)


# ---------------------------------------------------------------------------
# Final summary print
# ---------------------------------------------------------------------------


def print_final_summary(run_dir: Path, t_start: float, log) -> None:
    elapsed = time.time() - t_start
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)

    banner(f"EXPERIMENT COMPLETE  ({hours}h {mins}m {secs}s)", char="*")
    print(f"  Run directory : {run_dir}")
    print(f"  Results       : {run_dir / 'results'}")
    print(f"  Figures       : {run_dir / 'figures'}")
    print(f"  Log           : {run_dir / 'experiment.log'}")

    # Print accuracy summary from result files
    results_dir = run_dir / "results"
    summary_path = results_dir / "phase4_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print("\n  === Key results ===")

        def _acc(d, key="within_acc"):
            return d.get(key) if d else None

        rows = [
            ("Baseline A: CSP+LDA", summary.get("baselines", {}).get("csp_lda")),
            ("Baseline B: Riemannian", summary.get("baselines", {}).get("riemannian")),
            ("Baseline C: ViT-only", summary.get("baselines", {}).get("vit_only")),
            ("Dual-Branch (attention)", summary.get("dual_branch", {}).get("attention")),
            ("Transfer: EEG-Pretrain", summary.get("transfer_learning", {}).get("transfer")),
        ]
        for name, d in rows:
            acc = _acc(d)
            std = d.get("within_std") if d else None
            if acc is not None:
                print(f"  {name:<35}  {acc:6.2f}% ± {std or 0:5.2f}%")
            else:
                print(f"  {name:<35}  (not available)")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full overnight experiment runner for BCI thesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Resume from an existing run directory (skips completed stages). "
        "If omitted, a new timestamped directory is created.",
    )
    p.add_argument("--data-dir", default="~/mne_data", help="MNE data directory")
    p.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5, help="CV folds (within-subject)")
    p.add_argument("--epochs", type=int, default=50, help="Max training epochs per fold")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--pretrain-subjects",
        type=int,
        default=None,
        help="Max PhysioNet subjects for pretraining (None = all 109)",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Repetitions per fraction in reduced-data experiment",
    )
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
        help="Training-set fractions for reduced-data experiment",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Stage 1 dataset verification (assume data is present)",
    )
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip Stages 2-4 (CSP/Riemannian/ViT baselines)",
    )
    p.add_argument(
        "--skip-dual-branch",
        action="store_true",
        help="Skip Stages 5-6 (dual-branch fusion ablation)",
    )
    p.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Skip Stages 7-9 (pretrain / finetune / reduced-data)",
    )
    p.add_argument(
        "--skip-phase4",
        action="store_true",
        help="Skip Stage 10 (compile / visualize / stats)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Download data and print the plan, then exit without training.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- Create run directory ---
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Resuming run in: {run_dir}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path("runs") / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"New run directory: {run_dir}")

    log = setup_logging(run_dir / "experiment.log")
    log.info("Run directory: %s", run_dir)

    # --- Device ---
    from bci.utils.seed import get_device, set_seed

    device = get_device(args.device)
    log.info("Device: %s", device)

    t_start = time.time()

    TOTAL_STAGES = 10

    # ── Stage 1: Download ─────────────────────────────────────────────────
    stage_n = 1
    stage_banner(stage_n, TOTAL_STAGES, "Download / verify datasets")
    if args.skip_download:
        log.info("Stage 1 skipped (--skip-download).")
    else:
        stage_download(args, run_dir, log)

    if args.dry_run:
        banner("DRY RUN: stopping after data verification.")
        log.info("Dry run complete. Exiting.")
        return

    # Load BCI IV-2a data once (shared across all stages that need it)
    log.info("Loading BCI IV-2a subject data (shared for all stages)...")
    subject_data = load_bci_iv2a(args.data_dir, logger=log)
    if not subject_data:
        log.error("No BCI IV-2a data available. Aborting.")
        sys.exit(1)

    # ── Stage 2: Baseline A ───────────────────────────────────────────────
    stage_n = 2
    stage_banner(stage_n, TOTAL_STAGES, "Baseline A: CSP + LDA")
    if not args.skip_baselines:
        stage_baseline_a(subject_data, run_dir, args.n_folds, args.seed, log)
    else:
        log.info("Stage 2 skipped (--skip-baselines).")

    # ── Stage 3: Baseline B ───────────────────────────────────────────────
    stage_n = 3
    stage_banner(stage_n, TOTAL_STAGES, "Baseline B: Riemannian + LDA")
    if not args.skip_baselines:
        stage_baseline_b(subject_data, run_dir, args.n_folds, args.seed, log)
    else:
        log.info("Stage 3 skipped (--skip-baselines).")

    # ── Stage 4: Baseline C ───────────────────────────────────────────────
    stage_n = 4
    stage_banner(stage_n, TOTAL_STAGES, "Baseline C: CWT + ViT-Tiny")
    if not args.skip_baselines:
        stage_baseline_c(
            subject_data,
            run_dir,
            args.n_folds,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
            log,
        )
    else:
        log.info("Stage 4 skipped (--skip-baselines).")

    # ── Stage 5: Dual-branch attention (within + LOSO) ───────────────────
    stage_n = 5
    stage_banner(stage_n, TOTAL_STAGES, "Phase 2: Dual-branch, attention fusion (within + LOSO)")
    if not args.skip_dual_branch:
        stage_dual_branch(
            "attention",
            "within_subject",
            subject_data,
            run_dir,
            args.n_folds,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
            log,
        )
        stage_dual_branch(
            "attention",
            "loso",
            subject_data,
            run_dir,
            args.n_folds,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
            log,
        )
    else:
        log.info("Stage 5 skipped (--skip-dual-branch).")

    # ── Stage 6: Dual-branch gated ────────────────────────────────────────
    stage_n = 6
    stage_banner(stage_n, TOTAL_STAGES, "Phase 2: Dual-branch, gated fusion")
    if not args.skip_dual_branch:
        stage_dual_branch(
            "gated",
            "within_subject",
            subject_data,
            run_dir,
            args.n_folds,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
            log,
        )
    else:
        log.info("Stage 6 skipped (--skip-dual-branch).")

    # ── Stage 7: Pretrain ViT on PhysioNet ────────────────────────────────
    stage_n = 7
    stage_banner(stage_n, TOTAL_STAGES, "Phase 3a: Pretrain ViT on PhysioNet")
    if not args.skip_transfer:
        checkpoint_path = stage_pretrain(
            run_dir,
            args.pretrain_subjects,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
            log,
        )
    else:
        checkpoint_path = run_dir / "checkpoints" / "vit_pretrained_physionet.pt"
        log.info("Stage 7 skipped (--skip-transfer). Checkpoint: %s", checkpoint_path)

    # ── Stage 8: Finetune comparison ──────────────────────────────────────
    stage_n = 8
    stage_banner(
        stage_n, TOTAL_STAGES, "Phase 3b: Finetune comparison (scratch / imagenet / transfer)"
    )
    if not args.skip_transfer:
        if not checkpoint_path.exists():
            log.warning(
                "Checkpoint not found (%s); transfer condition will fail. "
                "Running scratch + imagenet only.",
                checkpoint_path,
            )
            conditions = ["scratch", "imagenet"]
        else:
            conditions = ["scratch", "imagenet", "transfer"]
        stage_finetune(
            subject_data,
            run_dir,
            checkpoint_path,
            conditions,
            args.n_folds,
            args.epochs,
            args.batch_size,
            device,
            args.seed,
            log,
        )
    else:
        log.info("Stage 8 skipped (--skip-transfer).")

    # ── Stage 9: Reduced-data experiment ─────────────────────────────────
    stage_n = 9
    stage_banner(stage_n, TOTAL_STAGES, "Phase 3c: Reduced-data experiment")
    if not args.skip_transfer:
        if not checkpoint_path.exists():
            log.warning("Checkpoint not found; skipping reduced-data transfer condition.")
        else:
            stage_reduced_data(
                subject_data,
                run_dir,
                checkpoint_path,
                args.fractions,
                args.n_repeats,
                args.n_folds,
                args.epochs,
                args.batch_size,
                device,
                args.seed,
                log,
            )
    else:
        log.info("Stage 9 skipped (--skip-transfer).")

    # ── Stage 10: Phase 4 analysis ────────────────────────────────────────
    stage_n = 10
    stage_banner(stage_n, TOTAL_STAGES, "Phase 4: Compile + visualize + stats")
    if not args.skip_phase4:
        stage_phase4(run_dir, log)
    else:
        log.info("Stage 10 skipped (--skip-phase4).")

    # --- Final summary ---
    print_final_summary(run_dir, t_start, log)


if __name__ == "__main__":
    main()
