"""Stage 9 – Finetune comparison: scratch / imagenet / transfer.

Trains the full DualBranchModel under three initialisation conditions:
  scratch   – random weights
  imagenet  – ViT pretrained on ImageNet (frozen backbone)
  transfer  – ViT pretrained on PhysioNet EEG (Stage 8 checkpoint, frozen backbone)

Requires the Stage 8 checkpoint for the 'transfer' condition.

Output:
  <run-dir>/results/real_finetune_scratch.json
  <run-dir>/results/real_finetune_imagenet.json
  <run-dir>/results/real_finetune_transfer.json

Usage::

    uv run python scripts/pipeline/stage_09_finetune.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_09_finetune.py --run-dir runs/my_run \\
        --checkpoint runs/my_run/checkpoints/vit_pretrained_physionet.pt \\
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
        description="Stage 9: Finetune comparison (scratch / imagenet / transfer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",    required=True)
    p.add_argument("--data-dir",   default="~/mne_data")
    p.add_argument("--device",     default="auto")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds",    type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to ViT checkpoint from Stage 8 "
             "(default: <run-dir>/checkpoints/vit_pretrained_physionet.pt)",
    )
    p.add_argument(
        "--conditions", nargs="+",
        default=["scratch", "imagenet", "transfer"],
        choices=["scratch", "imagenet", "transfer"],
    )
    return p.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout)
    for lib in ("mne", "pyriemann", "timm", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_09")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "stage_09_finetune.log")
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

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else run_dir / "checkpoints" / "vit_pretrained_physionet.pt"
    )

    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader

    from bci.data.dual_branch_builder import DualBranchFoldBuilder
    from bci.models.dual_branch import DualBranchModel
    from bci.training.cross_validation import CVResult, FoldResult
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import get_device, set_seed

    MODEL_NAME = "DualBranch-Transfer"
    FUSION     = "attention"

    device  = get_device(args.device)
    log.info("Device: %s", device)
    _device = torch.device(device)

    subject_data = load_bci_iv2a(args.data_dir, log)
    if not subject_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    builder = DualBranchFoldBuilder(
        csp_n_components=6, csp_reg="ledoit_wolf",
        riemann_estimator="lwf", riemann_metric="riemann",
        sfreq=128.0, channel_names=["C3", "Cz", "C4"],
    )

    def build_model(condition, math_input_dim):
        use_imagenet = condition in ("imagenet", "transfer")
        cfg = ModelConfig(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=use_imagenet,
            vit_drop_rate=0.1, csp_n_components=6,
            math_hidden_dims=[256, 128], math_drop_rate=0.3,
            fusion_method=FUSION, fused_dim=128,
            classifier_hidden_dim=64, n_classes=2,
        )
        model = DualBranchModel(math_input_dim=math_input_dim, config=cfg)
        if condition == "transfer":
            if not checkpoint_path.exists():
                log.warning("Checkpoint not found at %s; skipping weight load.", checkpoint_path)
            else:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                backbone_state = {k: v for k, v in ckpt.items()
                                  if not k.startswith("backbone.head")}
                model.vit_branch.backbone.load_state_dict(backbone_state, strict=False)
            model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        elif condition == "imagenet":
            model.freeze_vit_backbone(unfreeze_last_n_blocks=2)
        return model

    for condition in args.conditions:
        out_path = run_dir / "results" / f"real_finetune_{condition}.json"
        if out_path.exists():
            log.info("Already exists: %s – skipping.", out_path)
            continue

        log.info("=== Condition: %s ===", condition.upper())
        all_folds: list[FoldResult] = []
        fold_counter = 0
        t0 = time.time()

        for sid, (X, y) in sorted(subject_data.items()):
            log.info("  Subject %d (%d trials)...", sid, len(y))
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                  random_state=args.seed)
            for train_idx, test_idx in skf.split(X, y):
                set_seed(args.seed + fold_counter)
                train_ds, test_ds, math_input_dim = builder.build_fold(
                    X[train_idx], y[train_idx], X[test_idx], y[test_idx]
                )
                model = build_model(condition, math_input_dim)

                def fwd(batch, _m=model):
                    imgs, feats, labels = batch
                    return _m(imgs.to(_device), feats.to(_device)), labels.to(_device)

                trainer = Trainer(
                    model=model, device=device,
                    learning_rate=1e-4, weight_decay=1e-4,
                    epochs=args.epochs, batch_size=args.batch_size,
                    warmup_epochs=5, patience=10,
                    label_smoothing=0.1, val_fraction=0.2,
                    seed=args.seed, num_workers=0,
                )
                trainer.fit(train_ds, forward_fn=fwd,
                            model_tag=f"{condition}_f{fold_counter}")

                test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2,
                                         shuffle=False, num_workers=0)
                y_pred, y_prob = trainer.predict(test_loader, forward_fn=fwd)
                m = compute_metrics(y[test_idx], y_pred, y_prob)
                fr = FoldResult(
                    fold=fold_counter, subject=sid,
                    accuracy=m["accuracy"], kappa=m["kappa"], f1_macro=m["f1_macro"],
                    n_train=len(train_idx), n_test=len(test_idx),
                    y_true=y[test_idx], y_pred=y_pred, y_prob=y_prob,
                )
                log.info("  [%s] Fold %d S%02d: acc=%.2f%%  kappa=%.3f",
                         condition.upper(), fold_counter, sid, fr.accuracy, fr.kappa)
                all_folds.append(fr)
                fold_counter += 1

        result  = CVResult(strategy="within_subject",
                           model_name=f"{MODEL_NAME}-{condition}",
                           folds=all_folds)
        elapsed = time.time() - t0
        log.info("[%s] Done in %.1fs: %.2f%% ± %.2f%%",
                 condition.upper(), elapsed, result.mean_accuracy, result.std_accuracy)

        data = {
            "model": f"{MODEL_NAME}-{condition}",
            "condition": condition, "strategy": "within_subject",
            "mean_accuracy": result.mean_accuracy, "std_accuracy": result.std_accuracy,
            "mean_kappa": result.mean_kappa, "mean_f1": result.mean_f1,
            "n_folds": len(all_folds), "per_subject": result.per_subject_accuracy,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Saved: %s", out_path)

    log.info("Stage 9 complete.")


if __name__ == "__main__":
    main()
