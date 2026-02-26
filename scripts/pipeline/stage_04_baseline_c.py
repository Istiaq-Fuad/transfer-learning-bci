"""Stage 4 – Baseline C: CWT + ViT-Tiny (within-subject only).

Converts EEG epochs to Morlet CWT spectrograms (C3→R, Cz→G, C4→B) and
fine-tunes a ViT-Tiny classifier. Within-subject 5-fold CV only.

Output: <run-dir>/results/real_baseline_c_vit.json

Usage::

    uv run python scripts/pipeline/stage_04_baseline_c.py --run-dir runs/my_run
    uv run python scripts/pipeline/stage_04_baseline_c.py --run-dir runs/my_run \\
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
        description="Stage 4: Baseline C – CWT + ViT-Tiny.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",    required=True, help="Run directory (results saved here)")
    p.add_argument("--data-dir",   default="~/mne_data")
    p.add_argument("--device",     default="auto", help="auto | cpu | cuda | mps")
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
    log = logging.getLogger("stage_04")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "stage_04_baseline_c.log")
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

    out_path = run_dir / "results" / "real_baseline_c_vit.json"
    if out_path.exists():
        log.info("Result already exists at %s – skipping.", out_path)
        return

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from bci.data.transforms import CWTSpectrogramTransform
    from bci.models.vit_branch import ViTBranch
    from bci.training.cross_validation import within_subject_cv_all
    from bci.training.trainer import Trainer
    from bci.utils.config import ModelConfig, SpectrogramConfig
    from bci.utils.seed import get_device, set_seed

    MODEL_NAME  = "CWT+ViT-Tiny"
    CHANNEL_NAMES = ["C3", "Cz", "C4"]
    SFREQ = 128.0

    device = get_device(args.device)
    log.info("Device: %s", device)

    spec_config = SpectrogramConfig(
        wavelet="morl", freq_min=4.0, freq_max=40.0,
        n_freqs=64, image_size=(224, 224), channel_mode="rgb_c3_cz_c4",
    )
    transform = CWTSpectrogramTransform(spec_config)

    def epochs_to_imgs(X):
        hwc = transform.transform_epochs(X, CHANNEL_NAMES, SFREQ)
        return hwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    _device = torch.device(device)

    def predict_fn(X_train, y_train, X_test):
        set_seed(args.seed)
        imgs_train = epochs_to_imgs(X_train)
        imgs_test  = epochs_to_imgs(X_test)
        train_ds = TensorDataset(
            torch.tensor(imgs_train),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.tensor(imgs_test),
            torch.tensor(np.zeros(len(X_test), dtype=np.int64), dtype=torch.long),
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2,
                                 shuffle=False, num_workers=0)

        model_config = ModelConfig(
            vit_model_name="vit_tiny_patch16_224", vit_pretrained=True,
            vit_drop_rate=0.1, n_classes=2,
        )
        model = ViTBranch(config=model_config, as_feature_extractor=False)
        model.freeze_backbone(unfreeze_last_n_blocks=2)

        trainer = Trainer(
            model=model, device=device,
            learning_rate=1e-4, weight_decay=1e-4,
            epochs=args.epochs, batch_size=args.batch_size,
            warmup_epochs=5, patience=10,
            label_smoothing=0.1, val_fraction=0.2,
            seed=args.seed, num_workers=0,
        )
        trainer.fit(train_ds, model_tag="baseline_c_fold")
        return trainer.predict(test_loader)

    subject_data = load_bci_iv2a(args.data_dir, log)
    if not subject_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    set_seed(args.seed)
    log.info("Within-subject %d-fold CV (ViT)...", args.n_folds)
    t0 = time.time()
    within = within_subject_cv_all(
        subject_data, predict_fn, model_name=MODEL_NAME,
        n_folds=args.n_folds, seed=args.seed,
    )
    log.info("Done in %.1fs: %.2f%% ± %.2f%%",
             time.time() - t0, within.mean_accuracy, within.std_accuracy)

    results = {
        "model": MODEL_NAME,
        "within_subject": {
            "mean_accuracy": within.mean_accuracy,
            "std_accuracy":  within.std_accuracy,
            "mean_kappa":    within.mean_kappa,
            "mean_f1":       within.mean_f1,
            "n_folds":       len(within.folds),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_path)
    log.info("Stage 4 complete.")


if __name__ == "__main__":
    main()
