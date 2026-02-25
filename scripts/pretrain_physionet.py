"""Phase 3 – Step 1: Pretrain ViT Branch on Source Dataset.

Pretrains only the ViT branch (as a standalone spectrogram classifier) on a
source dataset — either PhysioNet MMIDB (real data) or a large synthetic
population (when real data is unavailable).

The saved checkpoint contains the full ViTBranch state_dict and is loaded by
the fine-tuning script to initialise the ViT branch of DualBranchModel.

Architecture during pretraining:
    Input: CWT Spectrogram (224×224×3)
    -> ViT-Tiny backbone (ImageNet pretrained or random)
    -> Linear head (192 -> 2)
    -> Softmax -> Left / Right hand MI

Outputs:
    checkpoints/vit_pretrained_physionet.pt   (ViTBranch state_dict)
    results/pretrain_physionet.json           (pretraining metrics)

Usage:
    # Synthetic source pretraining (always works, no download needed)
    uv run python scripts/pretrain_physionet.py --data synthetic \\
        --n-subjects 20 --epochs 30 --no-pretrained

    # Real PhysioNet (requires download)
    uv run python scripts/pretrain_physionet.py --data real \\
        --n-subjects 30 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bci.data.transforms import CWTSpectrogramTransform
from bci.models.vit_branch import ViTBranch
from bci.training.cross_validation import make_synthetic_subject_data
from bci.training.evaluation import compute_metrics
from bci.training.trainer import Trainer
from bci.utils.config import ModelConfig, SpectrogramConfig
from bci.utils.seed import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("mne").setLevel(logging.ERROR)
logging.getLogger("pyriemann").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectrogram helpers
# ---------------------------------------------------------------------------

def _epochs_to_images(
    X: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    spec_config: SpectrogramConfig,
) -> np.ndarray:
    """Convert EEG epochs (n, ch, t) -> float32 CHW images (n, 3, H, W)."""
    transform = CWTSpectrogramTransform(spec_config)
    hwc = transform.transform_epochs(X, channel_names, sfreq)  # (n, H, W, 3) uint8
    return hwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0


def build_image_dataset(
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    spec_config: SpectrogramConfig,
) -> TensorDataset:
    """Build a TensorDataset of (image, label) pairs."""
    logger.info("  Generating CWT spectrograms for %d trials...", len(y))
    imgs = _epochs_to_images(X, channel_names, sfreq, spec_config)
    return TensorDataset(
        torch.tensor(imgs, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Real PhysioNet loader
# ---------------------------------------------------------------------------

def load_physionet_data(
    n_subjects: int | None,
    sfreq: float = 128.0,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str], float]:
    """Load PhysioNet MMIDB Left/Right MI data via MOABB.

    Returns:
        subject_data: dict of subject_id -> (X, y)
        channel_names: list of channel names
        sfreq: sampling frequency after resampling
    """
    import mne
    from moabb.datasets import PhysionetMI
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
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
            logger.info("  Subject %d: X=%s", sid, X.shape)
        except Exception as e:
            logger.warning("  Skipping subject %d: %s", sid, e)

    # Get channel names from first subject
    channel_names = ["C3", "Cz", "C4"]  # MOABB normalises names
    return subject_data, channel_names, sfreq


# ---------------------------------------------------------------------------
# Pretraining loop (subject-pooled split)
# ---------------------------------------------------------------------------

def pretrain_vit(
    subject_data: dict[int, tuple[np.ndarray, np.ndarray]],
    channel_names: list[str],
    sfreq: float,
    spec_config: SpectrogramConfig,
    device: str | torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    warmup_epochs: int,
    patience: int,
    use_imagenet_pretrained: bool,
    val_fraction: float,
    seed: int,
    checkpoint_path: Path,
) -> dict:
    """Pool all source subjects, train ViT classifier, save checkpoint.

    Returns a dict with pretraining metrics.
    """
    set_seed(seed)

    # Pool all subjects
    all_X = np.concatenate([X for X, _ in subject_data.values()], axis=0)
    all_y = np.concatenate([y for _, y in subject_data.values()], axis=0)
    logger.info(
        "Pooled source data: %d trials, classes=%s", len(all_y), np.unique(all_y)
    )

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_y))
    all_X, all_y = all_X[idx], all_y[idx]

    # Train / val split (for final reporting only; Trainer does internal val)
    n_val = max(1, int(len(all_y) * val_fraction))
    n_train = len(all_y) - n_val
    X_train, X_val = all_X[:n_train], all_X[n_train:]
    y_train, y_val = all_y[:n_train], all_y[n_train:]

    logger.info("  Train: %d  Val: %d", n_train, n_val)

    # Build datasets
    train_ds = build_image_dataset(X_train, y_train, channel_names, sfreq, spec_config)
    val_ds   = build_image_dataset(X_val, y_val, channel_names, sfreq, spec_config)

    # Build ViT classifier (NOT as feature extractor — has classification head)
    model_config = ModelConfig(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=use_imagenet_pretrained,
        vit_drop_rate=0.1,
        n_classes=2,
    )
    model = ViTBranch(config=model_config, as_feature_extractor=False)

    _device = torch.device(device)

    def forward_fn(batch):
        imgs, labels = batch
        imgs   = imgs.to(_device)
        labels = labels.to(_device)
        return model(imgs), labels

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

    t0 = time.time()
    trainer.fit(train_ds, forward_fn=forward_fn, model_tag="vit_pretrain")
    elapsed = time.time() - t0
    logger.info("Pretraining done in %.1fs", elapsed)

    # Evaluate on held-out val set
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred, y_prob = trainer.predict(val_loader, forward_fn=forward_fn)
    metrics = compute_metrics(y_val, y_pred, y_prob)
    logger.info(
        "Pretrain val accuracy: %.2f%%  kappa=%.3f",
        metrics["accuracy"], metrics["kappa"],
    )

    # Save ViT branch weights
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("ViT pretrained weights saved to %s", checkpoint_path)

    return {
        "val_accuracy": metrics["accuracy"],
        "val_kappa": metrics["kappa"],
        "val_f1": metrics["f1_macro"],
        "n_source_trials": int(len(all_y)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(checkpoint_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Step 1: Pretrain ViT on source dataset"
    )
    parser.add_argument(
        "--data",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Source dataset to pretrain on",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=20,
        help="Number of source subjects (synthetic: generates N subjects; real: first N)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train ViT from random weights (skip ImageNet init)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of pooled data held out for evaluation",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vit_pretrained_physionet.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--output",
        default="results/pretrain_physionet.json",
        help="Output JSON results path",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("Phase 3 – ViT Pretraining")
    logger.info("  Source data:   %s", args.data)
    logger.info("  N subjects:    %d", args.n_subjects)
    logger.info("  Epochs:        %d (patience=%d)", args.epochs, args.patience)
    logger.info("  LR:            %.2e", args.lr)
    logger.info("  Batch size:    %d", args.batch_size)
    logger.info("  Device:        %s", device)
    logger.info("  ImageNet init: %s", not args.no_pretrained)
    logger.info("=" * 60)

    # Spectrogram config
    spec_config = SpectrogramConfig(
        wavelet="morl", freq_min=4.0, freq_max=40.0,
        n_freqs=64, image_size=(224, 224), channel_mode="rgb_c3_cz_c4",
    )

    # Load source data
    if args.data == "synthetic":
        logger.info("Generating synthetic source data (%d subjects)...", args.n_subjects)
        subject_data = make_synthetic_subject_data(n_subjects=args.n_subjects)
        channel_names = ["C3", "Cz", "C4"]
        sfreq = 128.0
    else:
        logger.info("Loading PhysioNet MMIDB (%d subjects)...", args.n_subjects)
        subject_data, channel_names, sfreq = load_physionet_data(
            n_subjects=args.n_subjects, sfreq=128.0
        )

    if not subject_data:
        logger.error("No source data loaded. Exiting.")
        return

    logger.info("Loaded %d subjects.", len(subject_data))

    # Pretrain
    checkpoint_path = Path(args.checkpoint)
    metrics = pretrain_vit(
        subject_data=subject_data,
        channel_names=channel_names,
        sfreq=sfreq,
        spec_config=spec_config,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        use_imagenet_pretrained=not args.no_pretrained,
        val_fraction=args.val_fraction,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
    )

    # Save results
    results = {
        "phase": "pretrain",
        "source": args.data,
        "n_subjects": args.n_subjects,
        **metrics,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Pretrain val accuracy: %.2f%%", metrics["val_accuracy"])
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Results:    %s", out_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
