"""Stage 04 – Pretrain ViT backbone on PhysioNet MMIDB spectrograms.

Loads pre-cached 9-channel multichannel CWT spectrograms for all PhysioNet
subjects (written by Stage 01) and trains a ViT-Tiny classifier.  Saves
the resulting backbone weights as a checkpoint for use in Stages 05–06.

Prerequisite: Stage 01 must have been run first.

Output::

    <run-dir>/checkpoints/vit_pretrained_physionet_vit.pt
    <run-dir>/results/real_pretrain_physionet_vit.json
    <run-dir>/plots/stage_04_pretrain/  (training loss and val-accuracy curves)

Usage::

    uv run python scripts/pipeline/stage_04_pretrain_vit.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_04_pretrain_vit.py \\
        --run-dir runs/my_run --epochs 50 --batch-size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bci.utils.logging import setup_stage_logging


def pretrain_vit(
    subject_data: dict,
    channel_names: list[str],
    sfreq: float,
    spec_config,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 16,
    warmup_epochs: int = 2,
    patience: int = 10,
    use_imagenet_pretrained: bool = True,
    val_fraction: float = 0.2,
    seed: int = 0,
    checkpoint_path: Path | None = None,
) -> dict:
    """Pretrain a ViT backbone on spectrogram images built from raw EEG data.

    Builds CWT spectrograms on the fly from *subject_data*, trains a
    :class:`bci.models.vit_branch.ViTBranch` classifier, and saves the
    backbone weights to *checkpoint_path*.

    Parameters
    ----------
    subject_data:
        Mapping of subject_id -> (X, y) where X is (n_trials, n_channels, n_times).
    channel_names:
        Channel names corresponding to the second axis of X.
    sfreq:
        EEG sampling frequency in Hz.
    spec_config:
        :class:`bci.utils.config.SpectrogramConfig` instance controlling CWT
        and image-size parameters.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
    epochs:
        Maximum training epochs.
    lr:
        Initial learning rate.
    batch_size:
        Mini-batch size.
    warmup_epochs:
        Number of linear warm-up epochs for the LR scheduler.
    patience:
        Early-stopping patience in epochs.
    use_imagenet_pretrained:
        Whether to initialise the ViT backbone from ImageNet weights.
    val_fraction:
        Fraction of pooled data held out for validation.
    seed:
        Random seed for reproducibility.
    checkpoint_path:
        Where to save the backbone ``state_dict``.  If ``None`` a temporary
        file is used and the path is included in the returned metrics dict.

    Returns
    -------
    dict
        Metrics dictionary containing at least ``"val_accuracy"``.
    """
    import tempfile

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from bci.data.transforms import CWTSpectrogramTransform
    from bci.models.vit_branch import ViTBranch
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import set_seed

    set_seed(seed)
    _device = torch.device(device)

    # Build spectrograms from raw EEG
    transform = CWTSpectrogramTransform(config=spec_config)
    all_imgs, all_y = [], []
    for subj_id in sorted(subject_data.keys()):
        X, y = subject_data[subj_id]
        imgs = transform.transform_epochs(X, channel_names, sfreq)
        all_imgs.append(imgs)
        all_y.append(y)

    X_all = np.concatenate(all_imgs, axis=0).astype(np.float32)
    y_all = np.concatenate(all_y, axis=0)

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_all))
    X_all, y_all = X_all[idx], y_all[idx]

    n_val = max(1, int(len(y_all) * val_fraction))
    n_train = len(y_all) - n_val

    train_ds = TensorDataset(
        torch.from_numpy(X_all[:n_train]),
        torch.from_numpy(y_all[:n_train].astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_all[n_train:]),
        torch.from_numpy(y_all[n_train:].astype(np.int64)),
    )

    n_chans = X_all.shape[1]
    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=use_imagenet_pretrained,
        vit_drop_rate=0.1,
        in_chans=n_chans,
        n_classes=int(len(np.unique(y_all))),
    )
    model = ViTBranch(config=model_config, as_feature_extractor=False)

    def fwd(batch, _m=model):
        imgs, labels = batch
        return _m(imgs.to(_device)), labels.to(_device)

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=lr,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=batch_size,
        warmup_epochs=warmup_epochs,
        patience=patience,
        label_smoothing=0.1,
        val_fraction=0.2,
        seed=seed,
        num_workers=0,
    )
    trainer.fit(train_ds, forward_fn=fwd, model_tag="pretrain_vit")

    # Validation
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(val_loader, forward_fn=fwd)
    metrics = compute_metrics(y_all[n_train:], y_pred, y_prob)

    # Save checkpoint
    _tmp = None
    if checkpoint_path is None:
        _tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        checkpoint_path = Path(_tmp.name)
        _tmp.close()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), checkpoint_path)

    metrics["checkpoint"] = str(checkpoint_path)
    # Alias: tests and callers may expect "val_accuracy"
    metrics.setdefault("val_accuracy", metrics.get("accuracy", 0.0))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 04: Pretrain ViT backbone on PhysioNet spectrograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--processed-dir",
        default=None,
        help="Root of processed .npz cache (default: data/processed/)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Number of PhysioNet subjects to use (default: all cached)",
    )
    args, _ = p.parse_known_args()
    return args


def save_pretrain_plot(history: list, best_epoch: int, plots_dir: Path, tag: str) -> None:
    """Save training loss + val accuracy curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = [r.epoch for r in history]
    train_losses = [r.train_loss for r in history]
    val_losses = [r.val_loss for r in history]
    val_accs = [r.val_accuracy for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_range, train_losses, label="Train Loss")
    axes[0].plot(epochs_range, val_losses, label="Val Loss")
    axes[0].axvline(
        x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{tag} – Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, val_accs, label="Val Accuracy", color="orange")
    axes[1].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{tag} – Val Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / f"{tag}_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log = setup_stage_logging(run_dir, "stage_04", "stage_04_pretrain_vit.log")

    checkpoint_path = run_dir / "checkpoints" / "vit_pretrained_physionet_vit.pt"
    out_path = run_dir / "results" / "real_pretrain_physionet_vit.json"
    plots_dir = run_dir / "plots" / "stage_04_pretrain"

    if checkpoint_path.exists() and out_path.exists():
        log.info("Checkpoint and results already exist – skipping Stage 04.")
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    from bci.data.download import load_spectrogram_stats
    from bci.models.vit_branch import ViTBranch
    from bci.training.evaluation import compute_metrics
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig
    from bci.utils.seed import get_device, set_seed

    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    device_str = get_device(args.device)
    device = torch.device(device_str)
    log.info("Device: %s", device_str)
    set_seed(args.seed)

    # ── Load spectrogram cache ─────────────────────────────────────────────
    log.info("Loading PhysioNet spectrogram cache...")
    try:
        mean, std = load_spectrogram_stats("physionet", data_dir=processed_dir)
    except FileNotFoundError as e:
        log.error("%s  Run Stage 01 first.", e)
        sys.exit(1)

    # Discover available subject IDs
    from bci.data.download import _processed_dir as _get_processed_dir

    pdir = _get_processed_dir("physionet", processed_dir)
    spec_files = sorted(pdir.glob("subject_[0-9]*_spectrograms.npz"))
    if not spec_files:
        log.error("No PhysioNet spectrogram files found in %s. Run Stage 01 first.", pdir)
        sys.exit(1)

    subject_ids = [int(p.stem.split("_")[1]) for p in spec_files]
    if args.n_subjects is not None:
        subject_ids = subject_ids[: args.n_subjects]

    # ── Build a lazy dataset to avoid loading everything into RAM ──────────
    # Build an index: list of (npz_path, trial_idx, label) by scanning only
    # the label arrays (tiny) — the spectrogram images are loaded per-batch.

    class _LazySpectrogramDataset(Dataset):
        """Loads spectrogram trials on-the-fly from per-subject .npz files."""

        def __init__(
            self,
            index: list[tuple[Path, int, int]],
            mean: np.ndarray,
            std: np.ndarray,
        ):
            self._index = index  # (npz_path, trial_idx, label)
            self._mean = mean.reshape(1, -1, 1, 1)  # (1, C, 1, 1)
            self._std = std.reshape(1, -1, 1, 1)
            # Per-file cache: keep one subject's data loaded at a time to
            # amortise npz reads when the DataLoader samples consecutive
            # trials from the same file (which is common after shuffling
            # within subjects).
            self._cached_path: Path | None = None
            self._cached_images: np.ndarray | None = None

        def __len__(self) -> int:
            return len(self._index)

        def __getitem__(self, idx: int):
            path, trial_idx, label = self._index[idx]
            # Load the .npz only when switching to a different file
            if self._cached_path != path:
                data = np.load(path)
                self._cached_images = data["images"]  # (n_trials, 9, 224, 224) float32
                self._cached_path = path
            img = self._cached_images[trial_idx].astype(np.float32)
            # Normalise per-channel
            img = (img - self._mean[0]) / self._std[0]
            return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)

    log.info("Building trial index for %d subjects...", len(subject_ids))
    from bci.data.download import _spectrogram_cache_path

    all_index: list[tuple[Path, int, int]] = []
    for sid in subject_ids:
        spec_path = _spectrogram_cache_path("physionet", sid, processed_dir)
        if not spec_path.exists():
            log.warning("  Subject %d: spectrogram file missing, skipping", sid)
            continue
        # Only load the tiny label array to build the index
        data = np.load(spec_path)
        y = data["y"].astype(np.int32)
        for trial_idx in range(len(y)):
            all_index.append((spec_path, trial_idx, int(y[trial_idx])))

    if not all_index:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    # Shuffle index
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(all_index)).tolist()
    all_index = [all_index[i] for i in shuffled]

    n_total = len(all_index)
    n_val = max(1, int(n_total * args.val_fraction))
    n_train = n_total - n_val
    log.info("Pooled: %d train + %d val trials  (lazy loading)", n_train, n_val)

    train_ds = _LazySpectrogramDataset(all_index[:n_train], mean, std)
    val_ds = _LazySpectrogramDataset(all_index[n_train:], mean, std)

    # ── Model ──────────────────────────────────────────────────────────────
    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=True,  # ImageNet init
        vit_drop_rate=0.1,
        in_chans=9,
        n_classes=2,
    )
    model = ViTBranch(config=model_config, as_feature_extractor=False)

    def fwd(batch, _m=model):
        imgs, labels = batch
        return _m(imgs.to(device)), labels.to(device)

    trainer = Trainer(
        model=model,
        device=device_str,
        learning_rate=args.lr,
        weight_decay=1e-4,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        label_smoothing=0.1,
        val_fraction=0.2,
        seed=args.seed,
        num_workers=0,
    )

    t0 = time.time()
    train_result = trainer.fit(train_ds, forward_fn=fwd, model_tag="stage04_pretrain")
    elapsed = time.time() - t0
    log.info("Pretraining done in %.1fs", elapsed)

    # Validation metrics
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(val_loader, forward_fn=fwd)
    y_true_val = np.array([all_index[n_train + i][2] for i in range(n_val)])
    metrics = compute_metrics(y_true_val, y_pred, y_prob)
    log.info("Pretrain val: %.2f%%  kappa=%.3f", metrics["accuracy"], metrics["kappa"])

    # Plot curves
    try:
        save_pretrain_plot(
            train_result.history, train_result.best_epoch, plots_dir, "stage04_pretrain"
        )
        log.info("Plots saved: %s", plots_dir)
    except Exception as e:
        log.warning("Plot save failed: %s", e)

    # ── Save checkpoint ────────────────────────────────────────────────────
    # Save only the backbone weights (keys match what finetune stages expect to load)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), checkpoint_path)
    log.info("Checkpoint saved: %s", checkpoint_path)

    results = {
        "stage": "04_pretrain_vit",
        "backbone": "vit_tiny_patch16_224",
        "source": "physionet",
        "n_subjects": len(subject_ids),
        "n_source_trials": n_total,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "val_accuracy": metrics["accuracy"],
        "val_kappa": metrics["kappa"],
        "val_f1": metrics["f1_macro"],
        "best_epoch": train_result.best_epoch,
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(checkpoint_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_path)
    log.info("Stage 04 complete.")


if __name__ == "__main__":
    main()
