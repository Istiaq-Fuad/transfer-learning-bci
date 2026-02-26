"""Stage 5 – Dual-branch, attention fusion (within-subject + LOSO).

Trains the full DualBranchModel (ViT + CSP + Riemannian) with attention
fusion. Runs 5-fold within-subject CV and LOSO CV.

Output:
  <run-dir>/results/real_dual_branch_attention.json
  <run-dir>/results/real_dual_branch_attention_loso.json

Usage::

    uv run python scripts/pipeline/stage_05_dual_attention.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_05_dual_attention.py --run-dir runs/my_run \\
        --epochs 50 --batch-size 32 --n-folds 5 --device cuda
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
        description="Stage 5: Dual-branch with attention fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",    required=True)
    p.add_argument("--data-dir",   default="~/mne_data")
    p.add_argument("--device",     default="auto")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds",    type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout)
    for lib in ("mne", "pyriemann", "timm", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_05")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "stage_05_dual_attention.log")
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


def run_dual_branch(
    fusion: str,
    strategy: str,
    subject_data: dict,
    run_dir: Path,
    n_folds: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    log,
) -> Path:
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

    tag = f"real_dual_branch_{fusion}"
    if strategy == "loso":
        tag += "_loso"
    out_path = run_dir / "results" / f"{tag}.json"

    if out_path.exists():
        log.info("Already exists: %s – skipping.", out_path)
        return out_path

    MODEL_NAME = "DualBranch-ViT+CSP+Riemann"
    builder = DualBranchFoldBuilder(
        csp_n_components=6, csp_reg="ledoit_wolf",
        riemann_estimator="lwf", riemann_metric="riemann",
        sfreq=128.0, channel_names=["C3", "Cz", "C4"],
    )
    _device = torch.device(device)

    def train_eval(fold_idx, subject_id, X_train, y_train, X_test, y_test):
        set_seed(seed + fold_idx)
        train_ds, test_ds, math_input_dim = builder.build_fold(
            X_train, y_train, X_test, y_test
        )
        model_config = ModelConfig(
            vit_model_name="efficientnet_b0", vit_pretrained=True,
            vit_drop_rate=0.1, csp_n_components=6,
            math_hidden_dims=[256, 128], math_drop_rate=0.3,
            fusion_method=fusion, fused_dim=256,
            classifier_hidden_dim=128, n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=model_config)
        model.freeze_vit_backbone(unfreeze_last_n_blocks=2)

        def fwd(batch):
            imgs, feats, labels = batch
            return model(imgs.to(_device), feats.to(_device)), labels.to(_device)

        trainer = Trainer(
            model=model, device=device,
            learning_rate=1e-4, weight_decay=1e-4,
            epochs=epochs, batch_size=batch_size,
            warmup_epochs=5, patience=10,
            label_smoothing=0.1, val_fraction=0.2,
            seed=seed, num_workers=0,
            backbone_lr_scale=0.1,
        )
        trainer.fit(train_ds, forward_fn=fwd, model_tag=f"dual_{fusion}_f{fold_idx}")

        test_loader = DataLoader(test_ds, batch_size=batch_size * 2,
                                 shuffle=False, num_workers=0)
        y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
        m = compute_metrics(y_test, y_pred, y_prob)
        fr = FoldResult(
            fold=fold_idx, subject=subject_id,
            accuracy=m["accuracy"], kappa=m["kappa"], f1_macro=m["f1_macro"],
            n_train=len(y_train), n_test=len(y_test),
            y_true=y_test, y_pred=y_pred, y_prob=y_prob,
        )
        sid_str = f"{subject_id:02d}" if subject_id is not None else "?"
        log.info("  Fold %d [S%s]: acc=%.2f%%  kappa=%.3f",
                 fold_idx, sid_str, fr.accuracy, fr.kappa)
        return fr

    t0 = time.time()
    all_folds: list[FoldResult] = []

    if strategy == "within_subject":
        fold_counter = 0
        for sid, (X, y) in sorted(subject_data.items()):
            log.info("Subject %d (%d trials)...", sid, len(y))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                fr = train_eval(fold_counter, sid,
                                X[train_idx], y[train_idx], X[test_idx], y[test_idx])
                all_folds.append(fr)
                fold_counter += 1
    else:
        subjects = sorted(subject_data.keys())
        for fold_idx, test_sid in enumerate(subjects):
            X_train = np.concatenate([subject_data[s][0] for s in subjects if s != test_sid])
            y_train = np.concatenate([subject_data[s][1] for s in subjects if s != test_sid])
            X_test, y_test = subject_data[test_sid]
            log.info("LOSO fold %d/%d: test=S%02d", fold_idx + 1, len(subjects), test_sid)
            fr = train_eval(fold_idx, test_sid, X_train, y_train, X_test, y_test)
            all_folds.append(fr)

    elapsed = time.time() - t0
    result = CVResult(strategy=strategy, model_name=MODEL_NAME, folds=all_folds)
    log.info("Done in %.1fs: %.2f%% ± %.2f%%",
             elapsed, result.mean_accuracy, result.std_accuracy)

    data = {
        "model": MODEL_NAME, "fusion": fusion, "strategy": strategy,
        "mean_accuracy": result.mean_accuracy, "std_accuracy": result.std_accuracy,
        "mean_kappa": result.mean_kappa, "mean_f1": result.mean_f1,
        "n_folds": len(all_folds), "per_subject": result.per_subject_accuracy,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved: %s", out_path)
    return out_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_logging(run_dir)

    from bci.utils.seed import get_device, set_seed
    device = get_device(args.device)
    log.info("Device: %s", device)

    subject_data = load_bci_iv2a(args.data_dir, log)
    if not subject_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    set_seed(args.seed)
    log.info("Running dual-branch [attention] within-subject CV...")
    run_dual_branch("attention", "within_subject", subject_data, run_dir,
                    args.n_folds, args.epochs, args.batch_size, device, args.seed, log)

    log.info("Running dual-branch [attention] LOSO CV...")
    run_dual_branch("attention", "loso", subject_data, run_dir,
                    args.n_folds, args.epochs, args.batch_size, device, args.seed, log)

    log.info("Stage 5 complete.")


if __name__ == "__main__":
    main()
