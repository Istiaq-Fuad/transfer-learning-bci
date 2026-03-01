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
    lp_epochs: int = 0,
    lp_lr: float = 1e-3,
    ft_lr: float = 3e-5,
    layer_lr_decay: float = 0.65,
    ft_weight_decay: float = 0.05,
) -> dict:
    """Pretrain a ViT backbone on spectrogram images built from raw EEG data.

    Builds CWT spectrograms on the fly from *subject_data*, trains a
    :class:`bci.models.vit_branch.ViTBranch` classifier, and saves the
    backbone weights to *checkpoint_path*.

    When *lp_epochs* > 0 the LP-FT (Linear Probing then Fine-Tuning)
    strategy is used:

    * **Phase 1 (LP)** – freeze backbone, train only the classification
      head for *lp_epochs* at *lp_lr*.
    * **Phase 2 (FT)** – unfreeze all layers, fine-tune with layer-wise
      LR decay for the remaining *epochs* at *ft_lr*.

    When *lp_epochs* == 0 (default) the original single-phase training
    at *lr* is used, keeping full backward compatibility.

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
        Maximum training epochs (used as-is in legacy mode, or as FT epochs
        in LP-FT mode).
    lr:
        Initial learning rate (legacy single-phase mode only).
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
    lp_epochs:
        Number of linear-probe epochs (0 to skip LP-FT and use legacy mode).
    lp_lr:
        Learning rate for the LP phase.
    ft_lr:
        Base learning rate for the FT phase.
    layer_lr_decay:
        Layer-wise LR decay factor for the FT phase.
    ft_weight_decay:
        Weight decay for the FT phase.

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

    if lp_epochs > 0:
        # -- LP-FT: two-phase training ------------------------------------
        # Phase 1: Linear Probe
        model.freeze_backbone(unfreeze_last_n_blocks=0)
        trainer_lp = Trainer(
            model=model,
            device=device,
            learning_rate=lp_lr,
            weight_decay=0.0,
            epochs=lp_epochs,
            batch_size=batch_size,
            warmup_epochs=1,
            patience=lp_epochs,
            label_smoothing=0.1,
            seed=seed,
            num_workers=0,
        )
        trainer_lp.fit(
            train_ds,
            forward_fn=fwd,
            model_tag="pretrain_vit_lp",
            val_dataset=val_ds,
        )

        # Phase 2: Fine-Tune with layer-wise LR decay
        model.unfreeze_all()
        final_trainer = Trainer(
            model=model,
            device=device,
            learning_rate=ft_lr,
            weight_decay=ft_weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            patience=patience,
            label_smoothing=0.1,
            seed=seed,
            num_workers=0,
            layer_lr_decay=layer_lr_decay,
        )
        final_trainer.fit(
            train_ds,
            forward_fn=fwd,
            model_tag="pretrain_vit_ft",
            val_dataset=val_ds,
        )
    else:
        # -- Legacy: single-phase uniform-LR training ----------------------
        final_trainer = Trainer(
            model=model,
            device=device,
            learning_rate=lr,
            weight_decay=1e-4,
            epochs=epochs,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            patience=patience,
            label_smoothing=0.1,
            seed=seed,
            num_workers=0,
        )
        final_trainer.fit(
            train_ds,
            forward_fn=fwd,
            model_tag="pretrain_vit",
            val_dataset=val_ds,
        )

    # Validation
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = final_trainer.predict(val_loader, forward_fn=fwd)
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
    p.add_argument(
        "--lr", type=float, default=1e-4, help="(Legacy) uniform LR — only used when --lp-epochs=0"
    )
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
    # LP-FT hyperparameters
    p.add_argument(
        "--lp-epochs", type=int, default=5, help="Linear-probe phase epochs (0 to skip LP-FT)"
    )
    p.add_argument(
        "--lp-lr", type=float, default=1e-3, help="Learning rate for the linear-probe phase"
    )
    p.add_argument(
        "--ft-lr", type=float, default=3e-5, help="Base learning rate for the fine-tune phase"
    )
    p.add_argument("--ft-epochs", type=int, default=45, help="Fine-tune phase epochs")
    p.add_argument(
        "--ft-warmup-epochs", type=int, default=3, help="Warmup epochs for the fine-tune phase"
    )
    p.add_argument(
        "--layer-lr-decay",
        type=float,
        default=0.65,
        help="Layer-wise LR decay factor (0-1); earlier layers get lower LR",
    )
    p.add_argument(
        "--ft-weight-decay", type=float, default=0.05, help="Weight decay for the fine-tune phase"
    )
    args, _ = p.parse_known_args()
    return args


def save_pretrain_plot(history: list, best_epoch: int, plots_dir: Path, title: str) -> None:
    """Save training loss + val accuracy curves."""
    from bci.utils.visualization import save_training_curves

    best_val_acc = max(r.val_accuracy for r in history)
    save_training_curves(
        history,
        best_epoch,
        best_val_acc,
        plots_dir,
        filename="pretrain_curves",
        title=title,
    )


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

    # ── Build lazy-loading dataset ────────────────────────────────────────
    # Instead of loading the full 8.9 GB spectrogram array into RAM (which
    # exceeds 16 GB when combined with model + DataLoader overhead), we use
    # a lazy-loading Dataset that reads per-subject .npz files on demand.
    # An LRU cache holds the most recently accessed subjects in memory so
    # consecutive accesses within a batch are fast.  Peak RAM ≈ model +
    # cache_size × ~20 MB/subject + batch tensors ≈ 1–2 GB.
    from collections import OrderedDict

    from bci.data.download import _spectrogram_cache_path

    # Scan trial counts per subject (cheap — only reads 'y' lengths)
    log.info("Scanning trial counts for %d subjects...", len(subject_ids))
    subject_info: list[tuple[Path, int]] = []  # (npz_path, n_trials)
    n_total = 0
    for sid in subject_ids:
        spec_path = _spectrogram_cache_path("physionet", sid, processed_dir)
        if not spec_path.exists():
            log.warning("  Subject %d: spectrogram file missing, skipping", sid)
            continue
        with np.load(spec_path) as data:
            n = len(data["y"])
        subject_info.append((spec_path, n))
        n_total += n

    if n_total == 0:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    log.info(
        "Found %d trials across %d subjects (%.1f GB uncompressed)",
        n_total,
        len(subject_info),
        n_total * 9 * 224 * 224 * 4 / 1e9,
    )

    # Build a global index: for each trial, store which subject file it
    # belongs to and its local index within that file.
    # trial_map[global_idx] = (subject_file_idx, local_trial_idx)
    trial_map: list[tuple[int, int]] = []
    y_all = np.empty(n_total, dtype=np.int32)
    offset = 0
    for file_idx, (spec_path, n) in enumerate(subject_info):
        with np.load(spec_path) as data:
            y_all[offset : offset + n] = data["y"].astype(np.int32)
        for local_idx in range(n):
            trial_map.append((file_idx, local_idx))
        offset += n

    # Shuffle and split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_fraction))
    n_train = n_total - n_val
    log.info("Split: %d train + %d val trials", n_train, n_val)

    # Target image size: 64×64 gives ViT-Tiny 16 patches instead of 196 at
    # 224×224 — ~12× fewer attention operations, fits easily on 4 GB VRAM.
    # We resize on load so Stage 01 does not need to be re-run.
    TARGET_IMG_SIZE = 64

    log.info(
        "Input size: %dx%d (resized from 224x224 on load, %d patches per image)",
        TARGET_IMG_SIZE,
        TARGET_IMG_SIZE,
        (TARGET_IMG_SIZE // 16) ** 2,
    )

    class _LazySpectrogramDataset(Dataset):
        """Lazy-loading spectrogram dataset that reads .npz files on demand.

        Keeps an LRU cache of the most recently accessed subject files so
        that consecutive samples from the same subject (common within a batch)
        don't trigger repeated disk reads.  Peak RAM is bounded by cache_size
        × (subject data size) rather than the full 8.9 GB.

        Images are resized from the on-disk size (224×224) to TARGET_IMG_SIZE
        (64×64) at load time, keeping VRAM usage and forward-pass time low.

        NOTE: num_workers must be 0 — the LRU OrderedDict is not fork-safe
        and will deadlock if passed to multiprocessing DataLoader workers.
        """

        def __init__(
            self,
            indices: np.ndarray,
            trial_map: list[tuple[int, int]],
            subject_info: list[tuple[Path, int]],
            y_all: np.ndarray,
            mean: np.ndarray,
            std: np.ndarray,
            cache_size: int = 8,
            target_size: int = TARGET_IMG_SIZE,
        ):
            self._indices = indices
            self._trial_map = trial_map
            self._subject_info = subject_info
            self._y_all = y_all
            self._mean = mean.reshape(1, -1, 1, 1)  # (1, 9, 1, 1)
            self._std = std.reshape(1, -1, 1, 1)
            self._cache_size = cache_size
            self._target_size = target_size
            # LRU cache: file_idx -> images array (n_trials, 9, H, W)
            self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        def _load_subject(self, file_idx: int) -> np.ndarray:
            """Load, resize, and normalise a subject's spectrograms with LRU caching."""
            if file_idx in self._cache:
                self._cache.move_to_end(file_idx)
                return self._cache[file_idx]
            spec_path = self._subject_info[file_idx][0]
            with np.load(spec_path) as data:
                images = data["images"].astype(np.float32)  # (N, 9, H, W)
            # Resize from on-disk size to target size using bilinear interpolation.
            # torch.nn.functional.interpolate is the fastest option for batched
            # (N, C, H, W) float32 arrays without requiring an extra import.
            h, w = images.shape[2], images.shape[3]
            if h != self._target_size or w != self._target_size:
                t = torch.from_numpy(images)
                t = torch.nn.functional.interpolate(
                    t,
                    size=(self._target_size, self._target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                images = t.numpy()
            # Normalise: mean/std are computed at 224×224 but are channel-wise
            # statistics so they remain valid after spatial resize.
            # Re-slice mean/std to match resized spatial dims (they are (1,9,1,1)).
            images -= self._mean
            images /= self._std
            # Evict oldest if cache is full
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[file_idx] = images
            return images

        def __len__(self) -> int:
            return len(self._indices)

        def __getitem__(self, i: int):
            global_idx = int(self._indices[i])
            file_idx, local_idx = self._trial_map[global_idx]
            images = self._load_subject(file_idx)
            img = torch.from_numpy(images[local_idx].copy())
            label = torch.tensor(int(self._y_all[global_idx]), dtype=torch.long)
            return img, label

    train_ds = _LazySpectrogramDataset(
        perm[:n_train],
        trial_map,
        subject_info,
        y_all,
        mean,
        std,
        cache_size=8,
    )
    val_ds = _LazySpectrogramDataset(
        perm[n_train:],
        trial_map,
        subject_info,
        y_all,
        mean,
        std,
        cache_size=8,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    # Use img_size=64: ViT-Tiny patch16 at 64×64 = 16 patches vs 196 at 224×224.
    # ~12× fewer attention ops → fits on 4 GB VRAM, ~10× faster per epoch.
    # ImageNet weights are still a valid initialisation even with a different
    # spatial resolution; timm interpolates positional embeddings automatically.
    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=True,  # ImageNet init
        vit_drop_rate=0.1,
        in_chans=9,
        n_classes=2,
    )
    model = ViTBranch(config=model_config, as_feature_extractor=False, img_size=TARGET_IMG_SIZE)

    def fwd(batch, _m=model):
        imgs, labels = batch
        return _m(imgs.to(device)), labels.to(device)

    t0 = time.time()
    use_lpft = args.lp_epochs > 0
    final_trainer: Trainer  # will be assigned in the branch below

    if use_lpft:
        # ── Phase 1: Linear Probe (LP) ────────────────────────────────────
        # Freeze backbone, train only the classification head.
        # This aligns the randomly-initialised head with pretrained features
        # before fine-tuning, preventing feature distortion (LP-FT, Kumar
        # et al., ICLR 2022).
        log.info("=" * 60)
        log.info("Phase 1 – Linear Probe: %d epochs, lr=%.2e", args.lp_epochs, args.lp_lr)
        log.info("=" * 60)
        model.freeze_backbone(unfreeze_last_n_blocks=0)
        log.info(
            "Trainable params (LP): %d / %d",
            model.get_num_params(trainable_only=True),
            model.get_num_params(trainable_only=False),
        )

        trainer_lp = Trainer(
            model=model,
            device=device_str,
            learning_rate=args.lp_lr,
            weight_decay=0.0,  # no regularisation during LP
            epochs=args.lp_epochs,
            batch_size=args.batch_size,
            warmup_epochs=1,
            patience=args.lp_epochs,  # never early-stop during LP
            label_smoothing=0.1,
            seed=args.seed,
            num_workers=0,  # _LazySpectrogramDataset is not fork-safe
        )
        lp_result = trainer_lp.fit(
            train_ds,
            forward_fn=fwd,
            model_tag="stage04_lp",
            val_dataset=val_ds,
        )
        log.info(
            "LP done: best_val_acc=%.2f%% @ epoch %d",
            lp_result.best_val_accuracy,
            lp_result.best_epoch,
        )

        # ── Phase 2: Full Fine-Tune (FT) with layer-wise LR decay ────────
        log.info("=" * 60)
        log.info(
            "Phase 2 – Fine-Tune: %d epochs, lr=%.2e, layer_decay=%.2f, wd=%.3f",
            args.ft_epochs,
            args.ft_lr,
            args.layer_lr_decay,
            args.ft_weight_decay,
        )
        log.info("=" * 60)
        model.unfreeze_all()
        log.info(
            "Trainable params (FT): %d / %d",
            model.get_num_params(trainable_only=True),
            model.get_num_params(trainable_only=False),
        )

        trainer_ft = Trainer(
            model=model,
            device=device_str,
            learning_rate=args.ft_lr,
            weight_decay=args.ft_weight_decay,
            epochs=args.ft_epochs,
            batch_size=args.batch_size,
            warmup_epochs=args.ft_warmup_epochs,
            patience=args.patience,
            label_smoothing=0.1,
            seed=args.seed,
            num_workers=0,  # _LazySpectrogramDataset is not fork-safe
            layer_lr_decay=args.layer_lr_decay,
        )
        ft_result = trainer_ft.fit(
            train_ds,
            forward_fn=fwd,
            model_tag="stage04_ft",
            val_dataset=val_ds,
        )
        log.info(
            "FT done: best_val_acc=%.2f%% @ epoch %d",
            ft_result.best_val_accuracy,
            ft_result.best_epoch,
        )

        # Use FT result as the main result; combine histories for plotting
        train_result = ft_result
        combined_history = lp_result.history + ft_result.history
        # Re-number epochs sequentially for the combined plot
        for i, r in enumerate(combined_history):
            combined_history[i] = type(r)(
                epoch=i + 1,
                train_loss=r.train_loss,
                val_loss=r.val_loss,
                val_accuracy=r.val_accuracy,
                val_kappa=r.val_kappa,
                lr=r.lr,
            )
        plot_history = combined_history
        best_plot_epoch = args.lp_epochs + ft_result.best_epoch
        final_trainer = trainer_ft
    else:
        # ── Legacy: single-phase uniform-LR training ──────────────────────
        log.info("Single-phase training: %d epochs, lr=%.2e", args.epochs, args.lr)
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
            seed=args.seed,
            num_workers=0,  # _LazySpectrogramDataset is not fork-safe
        )
        train_result = trainer.fit(
            train_ds,
            forward_fn=fwd,
            model_tag="stage04_pretrain",
            val_dataset=val_ds,
        )
        plot_history = train_result.history
        best_plot_epoch = train_result.best_epoch
        final_trainer = trainer

    elapsed = time.time() - t0
    log.info("Pretraining done in %.1fs", elapsed)

    # Validation metrics
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = final_trainer.predict(val_loader, forward_fn=fwd)
    metrics = compute_metrics(y_all[perm[n_train:]], y_pred, y_prob)
    log.info("Pretrain val: %.2f%%  kappa=%.3f", metrics["accuracy"], metrics["kappa"])

    # Plot curves
    try:
        plot_title = (
            "ViT Pretraining on PhysioNet (LP-FT)" if use_lpft else "ViT Pretraining on PhysioNet"
        )
        save_pretrain_plot(plot_history, best_plot_epoch, plots_dir, plot_title)
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
        "strategy": "lp_ft" if use_lpft else "uniform",
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
    if use_lpft:
        results["lp_epochs"] = args.lp_epochs
        results["lp_lr"] = args.lp_lr
        results["lp_best_val_acc"] = lp_result.best_val_accuracy
        results["ft_epochs"] = args.ft_epochs
        results["ft_lr"] = args.ft_lr
        results["layer_lr_decay"] = args.layer_lr_decay
        results["ft_weight_decay"] = args.ft_weight_decay
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_path)
    log.info("Stage 04 complete.")


if __name__ == "__main__":
    main()
