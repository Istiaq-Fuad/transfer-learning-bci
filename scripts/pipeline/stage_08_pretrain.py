"""Stage 8 – Pretrain image backbone on PhysioNet MMIDB.

Trains the image backbone branch as a standalone classifier on pooled PhysioNet
Left/Right Motor Imagery data. Saves the resulting weights as a checkpoint
for use in Stages 9 and 10.

Supported backbones (--backbone):
  vit_tiny_patch16_224  (default)
  efficientnet_b0

Output:
  <run-dir>/checkpoints/vit_pretrained_physionet_<backbone_short>.pt
  <run-dir>/results/real_pretrain_physionet_<backbone_short>.json
  <run-dir>/plots/stage_08_<backbone_short>/  (pretraining loss/accuracy curves)

Usage::

    uv run python scripts/pipeline/stage_08_pretrain.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_08_pretrain.py --run-dir runs/my_run \\
        --backbone vit_tiny_patch16_224 \\
        --n-subjects 109 --epochs 50 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Canonical per-backbone constants – single source of truth lives in the library.
import bci.models.vit_branch as _vit_mod
import bci.models.efficientnet_branch as _eff_mod

_BACKBONE_SHORT = {
    _vit_mod.MODEL_NAME: _vit_mod.BACKBONE_SHORT,
    _eff_mod.MODEL_NAME: _eff_mod.BACKBONE_SHORT,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 8: Pretrain image backbone on PhysioNet MMIDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",     required=True)
    p.add_argument("--data-dir",    default="~/mne_data",
                   help="Ignored by this stage (PhysioNet is downloaded automatically)")
    p.add_argument("--device",      default="auto")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--n-folds",     type=int, default=5,
                   help="Ignored by this stage (pretraining uses a fixed val split)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n-subjects",  type=int, default=None,
                   help="Number of PhysioNet subjects to use (default: all 109)")
    p.add_argument(
        "--backbone",
        default="vit_tiny_patch16_224",
        choices=list(_BACKBONE_SHORT.keys()),
        help="timm backbone model name",
    )
    p.add_argument(
        "--data",
        default="real",
        choices=["real", "synthetic"],
        help="'real' downloads PhysioNet MMIDB; 'synthetic' uses generated data (fast, for smoke tests)",
    )
    return p.parse_args()


def setup_logging(run_dir: Path, log_name: str) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        stream=sys.stdout)
    for lib in ("mne", "pyriemann", "timm", "matplotlib", "moabb"):
        logging.getLogger(lib).setLevel(logging.ERROR)
    log = logging.getLogger("stage_08")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / log_name)
    fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    log.addHandler(fh)
    return log


def save_pretrain_plot(train_result, plots_dir: Path, tag: str) -> None:
    """Save training loss + val accuracy curves for the pretraining run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    history = train_result.history

    epochs_range = [r.epoch for r in history]
    train_losses = [r.train_loss for r in history]
    val_losses   = [r.val_loss   for r in history]
    val_accs     = [r.val_accuracy for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_range, train_losses, label="Train Loss")
    axes[0].plot(epochs_range, val_losses,   label="Val Loss")
    axes[0].axvline(x=train_result.best_epoch, color="green", linestyle="--",
                    alpha=0.7, label=f"Best epoch {train_result.best_epoch}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{tag} – Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, val_accs, label="Val Accuracy", color="orange")
    axes[1].axvline(x=train_result.best_epoch, color="green", linestyle="--",
                    alpha=0.7, label=f"Best: {train_result.best_val_accuracy:.1f}%")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{tag} – Val Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / f"{tag}_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args     = parse_args()
    run_dir  = Path(args.run_dir)
    backbone = args.backbone
    bshort   = _BACKBONE_SHORT.get(backbone, backbone)
    log      = setup_logging(run_dir, f"stage_08_pretrain_{bshort}.log")
    log.info("Backbone: %s  (short: %s)", backbone, bshort)

    checkpoint_path = run_dir / "checkpoints" / f"vit_pretrained_physionet_{bshort}.pt"
    out_path        = run_dir / "results"     / f"real_pretrain_physionet_{bshort}.json"
    plots_dir       = run_dir / "plots"       / f"stage_08_{bshort}"

    if checkpoint_path.exists() and out_path.exists():
        log.info("Checkpoint and results already exist – skipping Stage 8.")
        return

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from bci.data.transforms import CWTSpectrogramTransform
    from bci.models.vit_branch import ViTBranch
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig, SpectrogramConfig
    from bci.utils.seed import get_device, set_seed

    device = get_device(args.device)
    log.info("Device: %s", device)
    set_seed(args.seed)

    channel_names = ["C3", "Cz", "C4"]

    if args.data == "synthetic":
        # ── Synthetic source data ──────────────────────────────────────────
        log.info("Using synthetic data for pretraining (smoke-test mode).")
        from bci.training.cross_validation import make_synthetic_subject_data
        subject_data = make_synthetic_subject_data(n_subjects=5, seed=args.seed)
        all_X_list = [X for X, _ in subject_data.values()]
        all_y_list = [y for _, y in subject_data.values()]
    else:
        # ── Load PhysioNet ─────────────────────────────────────────────────
        import mne
        from moabb.datasets import PhysionetMI
        from moabb.paradigms import LeftRightImagery

        mne.set_log_level("ERROR")
        log.info("Loading PhysioNet MMIDB (n_subjects=%s)...", args.n_subjects)
        dataset  = PhysionetMI()
        paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=128.0)
        subjects = dataset.subject_list
        if args.n_subjects is not None:
            subjects = subjects[: args.n_subjects]

        all_X_list, all_y_list = [], []
        for sid in subjects:
            try:
                X, y_labels, _ = paradigm.get_data(dataset=dataset, subjects=[sid])
                classes = sorted(np.unique(y_labels))
                label_map = {c: i for i, c in enumerate(classes)}
                y = np.array([label_map[lb] for lb in y_labels], dtype=np.int64)
                all_X_list.append(X.astype(np.float32))
                all_y_list.append(y)
                log.info("  Subject %d: X=%s", sid, X.shape)
            except Exception as e:
                log.warning("  Subject %d skipped: %s", sid, e)

    if not all_X_list:
        log.error("No source data loaded. Exiting.")
        sys.exit(1)

    all_X = np.concatenate(all_X_list, axis=0)
    all_y = np.concatenate(all_y_list, axis=0)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(all_y))
    all_X, all_y = all_X[idx], all_y[idx]
    log.info("Pooled source: %d trials", len(all_y))

    # ── CWT spectrograms ───────────────────────────────────────────────────
    spec_config = SpectrogramConfig(
        wavelet="morl", freq_min=4.0, freq_max=40.0,
        n_freqs=64, image_size=(224, 224), channel_mode="rgb_c3_cz_c4",
    )
    transform = CWTSpectrogramTransform(spec_config)

    def to_imgs(X):
        hwc = transform.transform_epochs(X, channel_names, 128.0)
        return hwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    n_val   = max(1, int(len(all_y) * 0.15))
    n_train = len(all_y) - n_val
    log.info("Generating CWT spectrograms for %d train trials...", n_train)
    imgs_train = to_imgs(all_X[:n_train])
    imgs_val   = to_imgs(all_X[n_train:])

    train_ds = TensorDataset(
        torch.tensor(imgs_train),
        torch.tensor(all_y[:n_train], dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(imgs_val),
        torch.tensor(all_y[n_train:], dtype=torch.long),
    )

    # ── Model + training ───────────────────────────────────────────────────
    model_config = ModelConfig(
        vit_model_name=backbone, vit_pretrained=True,
        vit_drop_rate=0.1, n_classes=2,
    )
    model   = ViTBranch(config=model_config, as_feature_extractor=False)
    _device = torch.device(device)

    def fwd(batch):
        imgs, labels = batch
        return model(imgs.to(_device)), labels.to(_device)

    trainer = Trainer(
        model=model, device=device,
        learning_rate=1e-4, weight_decay=1e-4,
        epochs=args.epochs, batch_size=args.batch_size,
        warmup_epochs=5, patience=10,
        label_smoothing=0.1, val_fraction=0.2,
        seed=args.seed, num_workers=0,
    )
    t0 = time.time()
    train_result = trainer.fit(
        train_ds, forward_fn=fwd,
        model_tag=f"{bshort}_pretrain",
    )
    elapsed = time.time() - t0
    log.info("Pretraining done in %.1fs", elapsed)

    # Save pretraining curves plot
    try:
        save_pretrain_plot(train_result, plots_dir, f"pretrain_{bshort}")
        log.info("Plots saved: %s", plots_dir)
    except Exception as e:
        log.warning("Pretrain plot save failed: %s", e)

    # Validation metrics
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(val_loader, forward_fn=fwd)
    metrics = compute_metrics(all_y[n_train:], y_pred, y_prob)
    log.info("Pretrain val: %.2f%%  kappa=%.3f", metrics["accuracy"], metrics["kappa"])

    # ── Save checkpoint ────────────────────────────────────────────────────
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    log.info("Checkpoint saved: %s", checkpoint_path)

    results = {
        "phase": "pretrain",
        "backbone": backbone,
        "source": "real_physionet",
        "n_subjects": len(all_X_list),
        "val_accuracy": metrics["accuracy"],
        "val_kappa":    metrics["kappa"],
        "val_f1":       metrics["f1_macro"],
        "n_source_trials": int(len(all_y)),
        "n_train": int(n_train),
        "n_val":   int(n_val),
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(checkpoint_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_path)
    log.info("Stage 8 complete.")


if __name__ == "__main__":
    main()
