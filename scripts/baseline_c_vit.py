"""Baseline C: CWT Spectrogram + ViT-Tiny classification pipeline.

Trains a standalone ViT-Tiny on CWT spectrogram images (224x224x3) using
within-subject k-fold cross-validation.

Pipeline:
    EEG epochs (n_trials, n_channels, n_times)
    -> CWT Morlet spectrogram  (C3→R, Cz→G, C4→B)
    -> ViT-Tiny (pretrained ImageNet, fine-tuned)
    -> Binary classifier (Left / Right hand MI)

For within-subject CV:
    - Each fold trains a fresh ViT from ImageNet weights
    - Uses AdamW + cosine LR schedule with linear warmup
    - Early stopping on validation accuracy

Usage:
    # Synthetic data (no download required)
    uv run python scripts/baseline_c_vit.py

    # Faster test (fewer epochs, no pretrained weights)
    uv run python scripts/baseline_c_vit.py --epochs 10 --no-pretrained

    # Real data
    uv run python scripts/baseline_c_vit.py --data real --data-dir ~/mne_data
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from bci.data.transforms import CWTSpectrogramTransform
from bci.models.vit_branch import ViTBranch
from bci.training.cross_validation import (
    CVResult,
    FoldResult,
    make_synthetic_subject_data,
    within_subject_cv_all,
)
from bci.training.evaluation import compute_metrics
from bci.training.trainer import Trainer
from bci.utils.config import ModelConfig, SpectrogramConfig
from bci.utils.seed import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("mne").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

MODEL_NAME = "CWT+ViT-Tiny"

# BCI IV-2a motor cortex channels (3 key channels for RGB spectrogram)
_C3_CZ_C4 = ["C3", "Cz", "C4"]
_SFREQ = 128.0  # after preprocessing


def build_spectrogram_transform(image_size: int = 224, n_freqs: int = 64) -> CWTSpectrogramTransform:
    """Build the CWT spectrogram transform used for this baseline."""
    config = SpectrogramConfig(
        wavelet="morl",
        freq_min=4.0,
        freq_max=40.0,
        n_freqs=n_freqs,
        image_size=(image_size, image_size),
        channel_mode="rgb_c3_cz_c4",
    )
    return CWTSpectrogramTransform(config)


def epochs_to_spectrograms(
    X: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    transform: CWTSpectrogramTransform,
    image_size: int = 224,
) -> np.ndarray:
    """Convert EEG epochs to CWT spectrogram images.

    Args:
        X: EEG epoch data (n_trials, n_channels, n_times).
        channel_names: Names for each channel.
        sfreq: Sampling frequency in Hz.
        transform: CWTSpectrogramTransform instance.
        image_size: Target image size.

    Returns:
        uint8 images of shape (n_trials, 3, image_size, image_size) as float32 [0,1].
    """
    # images: (n_trials, H, W, 3) uint8
    images_hwc = transform.transform_epochs(X, channel_names, sfreq)
    # -> (n_trials, 3, H, W) float32 in [0, 1]
    images_chw = images_hwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return images_chw


def make_vit_predict_fn(
    image_size: int,
    n_freqs: int,
    channel_names: list[str],
    sfreq: float,
    device: str | torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    warmup_epochs: int,
    patience: int,
    pretrained: bool,
    seed: int,
    unfreeze_last_n: int,
    checkpoint_dir: Path | None,
):
    """Return a predict_fn for CWT + ViT-Tiny.

    Each fold trains a fresh model from ImageNet weights.
    """
    transform = build_spectrogram_transform(image_size=image_size, n_freqs=n_freqs)

    def predict_fn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        set_seed(seed)

        # Convert epochs to spectrograms
        logger.info("    Converting %d train + %d test epochs to spectrograms...",
                    len(X_train), len(X_test))
        imgs_train = epochs_to_spectrograms(X_train, channel_names, sfreq, transform, image_size)
        imgs_test = epochs_to_spectrograms(X_test, channel_names, sfreq, transform, image_size)

        # PyTorch datasets
        train_ds = TensorDataset(
            torch.tensor(imgs_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.tensor(imgs_test, dtype=torch.float32),
            torch.tensor(np.zeros(len(X_test), dtype=np.int64), dtype=torch.long),
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0
        )

        # Fresh ViT-Tiny model with classification head
        model_config = ModelConfig(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=pretrained,
            vit_drop_rate=0.1,
            n_classes=2,
        )
        model = ViTBranch(config=model_config, as_feature_extractor=False)

        # Freeze most of ViT, unfreeze last N blocks
        if pretrained:
            model.freeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n)

        # Train
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
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            num_workers=0,
        )
        trainer.fit(train_ds, model_tag="baseline_c_fold")

        # Inference on test set
        y_pred, y_prob = trainer.predict(test_loader)
        return y_pred, y_prob

    return predict_fn


def print_results_table(within_result: CVResult) -> None:
    print("\n" + "=" * 60)
    print(f"  BASELINE C: {MODEL_NAME}")
    print("=" * 60)
    print(f"\n{'Strategy':<25} {'Accuracy':>10} {'±':>4} {'Std':>8} {'Kappa':>8} {'F1':>8}")
    print("-" * 60)
    print(
        f"{'Within-Subject (k-fold)':<25} "
        f"{within_result.mean_accuracy:>9.2f}% "
        f"{'':>4} "
        f"{within_result.std_accuracy:>7.2f}% "
        f"{within_result.mean_kappa:>8.4f} "
        f"{within_result.mean_f1:>8.4f}"
    )
    print("=" * 60)

    if within_result.per_subject_accuracy:
        print("\nPer-subject accuracy:")
        for subj, acc in sorted(within_result.per_subject_accuracy.items()):
            bar = "#" * int(acc / 5)
            print(f"  S{subj:02d}: {acc:6.2f}%  {bar}")
    print()


def save_results(within_result: CVResult, output_path: Path) -> None:
    results = {
        "model": MODEL_NAME,
        "within_subject": {
            "mean_accuracy": within_result.mean_accuracy,
            "std_accuracy": within_result.std_accuracy,
            "mean_kappa": within_result.mean_kappa,
            "mean_f1": within_result.mean_f1,
            "n_folds": len(within_result.folds),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


def load_real_data(
    data_dir: str,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str], float]:
    """Load BCI IV-2a data and return (subject_data, channel_names, sfreq)."""
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import LeftRightImagery

    mne.set_log_level("ERROR")
    logger.info("Loading BCI IV-2a dataset from %s ...", data_dir)
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=4.0, fmax=40.0, resample=_SFREQ)

    subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    channel_names: list[str] = []
    sfreq = _SFREQ

    for subject_id in dataset.subject_list:
        try:
            X, y_labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
            classes = sorted(np.unique(y_labels))
            label_map = {c: i for i, c in enumerate(classes)}
            y = np.array([label_map[lbl] for lbl in y_labels], dtype=np.int64)
            subject_data[subject_id] = (X.astype(np.float32), y)
            logger.info("  Subject %d: X=%s", subject_id, X.shape)
        except Exception as e:
            logger.warning("  Skipping subject %d: %s", subject_id, e)

    # For real BCI IV-2a data, MOABB provides channel names in metadata
    # Default to the 3 RGB channels
    channel_names = _C3_CZ_C4
    return subject_data, channel_names, sfreq


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline C: CWT Spectrogram + ViT-Tiny")
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-dir", type=str, default="~/mne_data")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-subjects", type=int, default=9, help="Subjects (synthetic only)")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (must be 224 for ViT-Tiny patch16)",
    )
    parser.add_argument("--n-freqs", type=int, default=64)
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet weights")
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=2,
        help="Number of ViT blocks to unfreeze when using pretrained weights",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/baseline_c_vit.json")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    if args.image_size != 224:
        logger.error(
            "ViT-Tiny with patch16 requires 224x224 input. "
            "Got --image-size %d. Overriding to 224.",
            args.image_size,
        )
        args.image_size = 224

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("=" * 50)
    logger.info("Baseline C: CWT Spectrogram + ViT-Tiny")
    logger.info("  Data:       %s", args.data)
    logger.info("  Device:     %s", device)
    logger.info("  Pretrained: %s", not args.no_pretrained)
    logger.info("  Epochs:     %d (patience=%d)", args.epochs, args.patience)
    logger.info("  LR:         %.2e", args.lr)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  CV folds:   %d", args.n_folds)
    logger.info("  Seed:       %d", args.seed)
    logger.info("=" * 50)

    # --- Load data ---
    if args.data == "synthetic":
        logger.info("Generating synthetic data (%d subjects)...", args.n_subjects)
        subject_data = make_synthetic_subject_data(n_subjects=args.n_subjects)
        # Synthetic data has 22 channels; use channels 3,10,9 as C3/Cz/C4 proxies
        # (indices match the 22-channel BCI IV-2a montage)
        channel_names = _C3_CZ_C4
        sfreq = 128.0
    else:
        subject_data, channel_names, sfreq = load_real_data(args.data_dir)

    if not subject_data:
        logger.error("No data loaded. Exiting.")
        return

    predict_fn = make_vit_predict_fn(
        image_size=args.image_size,
        n_freqs=args.n_freqs,
        channel_names=channel_names,
        sfreq=sfreq,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        seed=args.seed,
        unfreeze_last_n=args.unfreeze_last_n,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    # --- Within-subject CV ---
    logger.info("\nRunning within-subject %d-fold CV...", args.n_folds)
    t0 = time.time()
    within_result = within_subject_cv_all(
        subject_data,
        predict_fn,
        model_name=MODEL_NAME,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    t_within = time.time() - t0
    logger.info("Within-subject CV done in %.1fs", t_within)

    print_results_table(within_result)
    save_results(within_result, Path(args.output))


if __name__ == "__main__":
    main()
